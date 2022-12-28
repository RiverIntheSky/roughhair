#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/math.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/microfacet.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-roughhair:

Rough hair material (:monosp:`roughhair`)
-----------------------------------------------------

.. pluginparameters::

 * - int_ior
   - |float| or |string|
   - Interior index of refraction specified numerically or using a known material name. (Default: keratin / 1.5490)
 * - ext_ior
   - |float| or |string|
   - Exterior index of refraction specified numerically or using a known material name.  (Default: air / 1.000277)
 * - eumelanin
   - |float|
   - Eumelanin concentration. (Default 1)
 * - pheomelanin
   - |float|
   - Eumelanin concentration. (Default 1)
 * - tilt
   - |float|
   - hair scale tilt. The tilt direction should point to the root of the hair (Default: -2 degrees)
 * - distribution
   - |string|
   - Specifies the type of microfacet normal distribution used to model the surface roughness.

     - :monosp:`beckmann`: Physically-based distribution derived from Gaussian random surfaces.
       This is the default.
     - :monosp:`ggx`: The GGX :cite:`Walter07Microfacet` distribution (also known as Trowbridge-Reitz
       :cite:`Trowbridge19975Average` distribution) was designed to better approximate the long
       tails observed in measurements of ground surfaces, which are not modeled by the Beckmann
       distribution.
 * - roughness
   - |float|
   - Specifies the roughness of the unresolved surface micro-geometry along the tangent and
     bitangent directions. When the Beckmann distribution is used, this parameter is equal to the
     *root mean square* (RMS) slope of the microfacets. (Default: 0.13)
 * - sample_visible
   - |bool|
   - Enables a sampling technique proposed by Heitz and D'Eon :cite:`Heitz1014Importance`, which
     focuses computation on the visible parts of the microfacet normal distribution, considerably
     reducing variance in some cases. (Default: |true|, i.e. use visible normal sampling)
 * - analytical
   - |bool|
   - Whether analytical integration should be used for GGX R lobe.
   - when set to true, \int D is evaluated, ignoring geometric term G
   - when set to false, \int DG is evaluated numerically using Simpson's rule



This plugin implements a microfacet-based hair scattering model.
The geometry data (semi-major and semi-minor axes) are read from
the intersection data stored in si.dn_du.
This is the implementation of the paper *A Microfacet-based Hair Scattering Model*
by Huang et al. [2022]

The following XML snippet describes a material definition for hair:

.. code-block:: xml
    :name: roughhair

    <bsdf type="roughhair">
        <string name="distribution" value="beckmann"/>
        <float name="tilt" value="-3"/>
        <float name="eumelanin" value="0.02"/>
        <float name="pheomelanin" value="1.5"/>
        <float name="roughness" value="0.15"/>
    </bsdf>

 */

template <typename Float, typename Spectrum>
class RoughHair : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Sampler, MicrofacetDistribution)
    static constexpr auto Pi        = math::Pi<Float>;
    static constexpr auto TwoPi     = math::TwoPi<Float>;
    static constexpr auto Inf       = math::Infinity<Float>;

    RoughHair(const Properties &props) : Base(props) {
	// Specifies the external index of refraction at the interface
	ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air");

	// Specifies the internal index of refraction at the interface
	ScalarFloat int_ior = lookup_ior(props, "int_ior", "keratin");

	if (int_ior < 0.f || ext_ior < 0.f || int_ior == ext_ior)
            Throw("The interior and exterior indices of "
                  "refraction must be positive and differ!");

        m_eta = int_ior / ext_ior;
	m_inv_eta = ext_ior / int_ior;

	// roughness
        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann")
                m_type = MicrofacetType::Beckmann;
            else if (distr == "ggx")
                m_type = MicrofacetType::GGX;
            else
                Throw("Specified an invalid distribution \"%s\", must be "
                      "\"beckmann\" or \"ggx\"!", distr.c_str());
        } else {
            m_type = MicrofacetType::GGX;
        }
	m_roughness = props.float_("roughness", 0.13f);

	m_sample_visible = props.bool_("sample_visible", true);

	m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide | BSDFFlags::Anisotropic;

	// shape parameters
	m_tilt = props.float_("tilt", -2.f) * Pi / 180.f;

	// derived parameters
	m_roughness_squared = sqr(m_roughness);
	m_tan_tilt = tan(m_tilt);

	// hair color
	m_eumelanin = props.float_("eumelanin", 1);
	m_pheomelanin = props.float_("pheomelanin", 1);

	auto pmgr = PluginManager::instance();
	Properties props_sampler("independent");
	props_sampler.set_int("sample_count", 4);
	m_sampler = static_cast<Sampler *>(pmgr->create_object<Sampler>(props_sampler));
    }

    /* returns sin_theta */
    MTS_INLINE Float sintheta(const Vector3f& w) const {
	return w.y();
    }

    /* returns cos_theta */
    MTS_INLINE Float costheta(const Vector3f& w) const {
        return sqrt(sqr(w.x()) + sqr(w.z()));
    }

    /* returns tan_theta */
    MTS_INLINE Float tantheta(const Vector3f& w) const {
	return sintheta(w) / costheta(w);
    }

    MTS_INLINE Float sinphi(const Vector3f& w) const {
	return w.x() / costheta(w);
    }

    MTS_INLINE Float cosphi(const Vector3f& w) const {
	return w.z() / costheta(w);
    }

    MTS_INLINE std::pair<Float, Float> sincosphi(const Vector3f& w) const {
	Float cos_theta = costheta(w);
	return {w.x() / cos_theta, w.z() / cos_theta};
    }

    /* extract theta coordinate from 3D direction
     * -pi < theta < pi */
    MTS_INLINE Float dir_theta(const Vector3f& w) const {
	return atan2(sintheta(w), costheta(w));
    }

    /* extract phi coordinate from 3D direction.
     * -pi < phi < pi
     * Assuming phi(wi) = 0 */
    MTS_INLINE Float dir_phi(const Vector3f& w) const {
	return atan2(w.x(), w.z());
    }

    /* extract theta and phi coordinate from 3D direction
     * -pi/2 < theta < pi/2, -pi < phi < pi
     * Assuming phi(wi) = 0 */
    MTS_INLINE std::pair<Float, Float> dir_sph(const Vector3f& w) const {
	return std::make_pair(dir_theta(w), dir_phi(w));
    }

    /* compute the vector direction given spherical coordinates */
    MTS_INLINE Vector3f sph_dir(Float theta, Float phi) const {
	auto [sin_theta, cos_theta] = sincos(theta);
	auto [sin_phi,   cos_phi]   = sincos(phi);
	return Vector3f(sin_phi * cos_theta, sin_theta, cos_phi * cos_theta);
    }

    /* get waveleingths of the ray */
    MTS_INLINE Spectrum get_spectrum(const SurfaceInteraction3f &si) const {
	Spectrum wavelengths;
	if constexpr (is_spectral_v<Spectrum>) {
	    wavelengths[0] = si.wavelengths[0]; wavelengths[1] = si.wavelengths[1];
	    wavelengths[2] = si.wavelengths[2]; wavelengths[3] = si.wavelengths[3];
	} else {
	    wavelengths[0] = 612.f; wavelengths[1] = 549.f; wavelengths[2] = 465.f;
	}

	return wavelengths;
    }

    /* pheomelanin absorption coefficient */
    MTS_INLINE Spectrum pheomelanin(const Spectrum &lambda) const {
	return 2.9e12f * pow(lambda, -4.75f); // adjusted relative to 0.1mm hair width
    }

    /* eumelanin absorption coefficient */
    MTS_INLINE Spectrum eumelanin(const Spectrum &lambda) const {
	return 6.6e8f * pow(lambda, -3.33f); // adjusted relative to 0.1mm hair width
    }

    /* get semi major axis, semi minor axis and squared eccentricity */
    MTS_INLINE std::tuple<Float, Float, Float> get_abe2(const SurfaceInteraction3f &si) const {
	return {si.dn_du.x(), si.dn_du.y(), si.dn_du.z()};
    }

    /* convert between gamma and phi */
    MTS_INLINE Float to_phi(Float gamma, Float a, Float b) const {
	auto [sin_gamma, cos_gamma] = sincos(gamma);
	return atan2(b * sin_gamma, a * cos_gamma);
    }

    MTS_INLINE Float to_gamma(Float phi, Float a, Float b) const {
	auto [sin_phi, cos_phi] = sincos(phi);
	return atan2(a * sin_phi, b * cos_phi);
    }

    MTS_INLINE Point2f to_point(Float gamma, Float a, Float b) const {
	auto [sg, cg] = sincos(gamma);
	return Point2f(a * sg, b * cg);
    }

    /* given theta and gamma, convert to vector */
    MTS_INLINE Vector3f sphg_dir(Float theta, Float gamma, Float a, Float b) const {
	auto [sin_theta, cos_theta] = sincos(theta);
	auto [sin_gamma,   cos_gamma]   = sincos(gamma);
	Float tan_gamma = sin_gamma / cos_gamma;
	Float tan_phi = b / a * tan_gamma;
	Float cos_phi = enoki::mulsign(rsqrt(sqr(tan_phi) + 1.f), cos_gamma);
	Float sin_phi = cos_phi * tan_phi;
	return Vector3f(sin_phi * cos_theta, sin_theta, cos_phi * cos_theta);
    }

    /* sample microfacets from a tilted mesonormal */
    std::pair<Normal3f, Float> sample_wh(const Vector3f &wi, const Normal3f &wm,
					 const MicrofacetDistribution &distr,
					 const Point2f &sample1) const {
        /* Coordinate transformation for microfacet sampling */
        Frame3f wm_frame;
	wm_frame.n = wm;
	wm_frame.s = cross(Normal3f(0.f, 1.f, 0.f), wm);
	wm_frame.t = cross(wm_frame.n, wm_frame.s);
	auto from_wm = Transform4f::to_frame(wm_frame);
	Normal3f wh_wm, wh;
	Vector3f wi_wm = from_wm.inverse() * wi;
	Float pdf;
	std::tie(wh_wm, pdf) = distr.sample(wi_wm, sample1);

	wh = from_wm * wh_wm;
	return {wh, pdf};
    }

    /// Smith's separable shadowing-masking approximation
    Float G(const Vector3f &wi, const Vector3f &wo, const Normal3f &m, const Normal3f &h) const {
        return smith_g1(wi, m, h) * smith_g1(wo, m, h);
    }

    /**
     * \brief Smith's shadowing-masking function for mesonormal
     *
     * \param v
     *     An arbitrary direction
     * \param m
     *     The macrofacet normal
     * \param h
     *     The microfacet normal
     */
    Float smith_g1(const Vector3f &v, const Normal3f &m, const Normal3f &h) const {
	Float cos_vm = dot(v, m),
	      tmp, result;
	if (m_type == MicrofacetType::Beckmann) {
	    tmp = abs(rcp(sqr(cos_vm)) - 1.f);
	    Float a_sqr = rcp(m_roughness_squared * tmp),
		a = sqrt(a_sqr);
	    /* Use a fast and accurate (<0.35% rel. error) rational
               approximation to the shadowing-masking function */
            result = select(a >= 1.6f, 1.f,
                            (3.535f * a + 2.181f * a_sqr) /
                            (1.f + 2.276f * a + 2.577f * a_sqr));
	} else {
	    result = 2.f * rcp(1.f + sqrt(m_roughness_squared * rcp(sqr(cos_vm)) + 1.f - m_roughness_squared));
	}
        /* Assume consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
	masked(result, dot(v, h) <= 0.f || cos_vm <= 0.f) = 0.f;
	return result;
    }

    /// Check cylinder intersection
    Float G_(const Vector3f &wi, const Vector3f &wo, const Normal3f &m, const Normal3f &h) const {
        return smith_g1_(wi, m, h) * smith_g1_(wo, m, h);
    }

    Float smith_g1_(const Vector3f &v, const Normal3f &m, const Normal3f &h) const {
	return (dot(v, h) > 0 && dot(v, m) > 0);
    }

    // smith_g1 / dot(v, m)
    Float smith_g1_visible(const Vector3f &v, const Normal3f &m, const Normal3f &h) const {
	Float cos_vm = dot(v, m),
	      result;
	if (m_type == MicrofacetType::Beckmann) {
	    result = smith_g1(v, m, h) / cos_vm;
	} else {
	result = 2.f * rcp(cos_vm + sqrt(m_roughness_squared + (1.f - m_roughness_squared) * sqr(cos_vm)));
	}
        /* Assume consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
	masked(result, dot(v, h) <= 0.f || cos_vm <= 0.f) = 0.f;
	return result;
    }

    // NDF
    Float D(const Normal3f &m, const Normal3f &h) const {
	Float cos_theta = dot(h, m),
   	      result;

	if (m_type == MicrofacetType::Beckmann) {
	    result = exp((1.f - rcp(sqr(cos_theta))) / m_roughness_squared) / (Pi * m_roughness_squared * sqr(sqr(cos_theta)));
	} else { // GGX
	    result = m_roughness_squared * rcp(Pi * sqr(1.f + (m_roughness_squared - 1.f) * sqr(cos_theta)));
	}

	// Prevent potential numerical issues in other stages of the model
        return select(result * cos_theta > 1e-20f, result, 0.f);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext & ctx,
					     const SurfaceInteraction3f & si,
					     Float sample1,
					     const Point2f & sample2,
					     Mask active) const override {
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);
        BSDFSample3f bs = zero<BSDFSample3f>();
	Mask active_r, active_tt, active_trt;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || none_or<false>(active)))
            return { bs, 0.f };

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type, m_roughness, m_sample_visible);

	// generate samples
	Float sample_lobe = sample1;
	Float sample_h = const_cast<Sampler&>(*m_sampler).next_1d(active);
	Point2f sample_h1 = sample2;
	Point2f sample_h2 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	Point2f sample_h3 = const_cast<Sampler&>(*m_sampler).next_2d(active);

	// get geometry data from surface interaction
	auto [a, b, e2] = get_abe2(si);

	auto [sin_phi_i, cos_phi_i] = sincosphi(si.wi);

	Float d_i = sqrt(1.f - e2 * sqr(sin_phi_i));
	Float h = d_i * (sample_h * 2.f - 1.f);
	Float gamma_mi = atan2(cos_phi_i, - b / a * sin_phi_i) - acos(h * rsqrt(sqr(cos_phi_i) + sqr(b / a * sin_phi_i)));
	auto [sin_gamma_mi, cos_gamma_mi] = sincos(gamma_mi);
	Normal3f wmi_ = normalize(Normal3f(b * sin_gamma_mi, 0.f, a * cos_gamma_mi)); /* macronormal */
	auto [st, ct] = sincos(m_tilt);
	Normal3f wmi(wmi_.x() * ct, st, wmi_.z() * ct); /* mesonormal */

	if (dot(wmi, si.wi) < 0 || dot(wmi_, si.wi) < 0)
	    return {bs, 0.f}; /* macro/mesonormal invisible */

        // sample R lobe
	auto [wh1, pdfh1] = sample_wh(si.wi, wmi, distr, sample_h1);
	Vector3f wr = reflect(si.wi, wh1);

	/* Ensure that this is a valid sample */
	active &= (dot(wr, wh1) > 0 && dot(wr, wmi) > 0 && G_(si.wi, wr, wmi_, wh1) > 0 && pdfh1 > 0);
	active_r = active;

	/* fresnel coefficient */
        auto [R1, cos_theta_t1, eta_it1, eta_ti1] = fresnel(dot(si.wi, wh1), Float(m_eta));
	Spectrum R = select(active_r, R1, 0.f);

	// sample TT lobe
	Vector3f wt = refract(si.wi, wh1, cos_theta_t1, eta_ti1);
	Float phi_t = dir_phi(wt);
	Float gamma_mt = 2.f * to_phi(phi_t, a, b) - gamma_mi;
	Vector3f wmt = sphg_dir(-m_tilt, gamma_mt, a, b);
	Vector3f wmt_ = sphg_dir(0, gamma_mt, a, b);
	auto [wh2, pdfh2] = sample_wh(-wt, wmt, distr, sample_h2);
	Vector3f wtr = reflect(wt, wh2);

	/* fresnel coefficient */
        auto [R2, cos_theta_t2, eta_it2, eta_ti2] = fresnel(dot(-wt, wh2), Float(m_inv_eta));

	Vector3f wtt = refract(-wt, wh2, cos_theta_t2, eta_ti2);
	active_tt = (active && dot(wt, wh2) < 0 && dot(wmt, wt) < 0 && pdfh2 > 0
		     && G_(-wt, -wtr, Normal3f(wmt.x(), 0.f, wmt.z()), wh2) > 0); // visibility
	active_trt = active_tt;
	active_tt &= (dot(wtt, wmt) < 0);
	active_tt &= (cos_theta_t2 != 0); // total internal reflection
	Spectrum T1 = 1.f - R1;
	Spectrum T2 = 1.f - R2;

	/* absorption */
	Spectrum wavelengths = get_spectrum(si);
	Spectrum mu_a = fmadd(m_pheomelanin, pheomelanin(wavelengths),
			      m_eumelanin * eumelanin(wavelengths));
	Point2f pi = to_point(gamma_mi, a, b);
	Point2f pt = to_point(gamma_mt + Pi, a, b);
	/* divide by 0.05 (* 20) because the absorption is defined for hair width 0.1 */
	Spectrum A_t = exp(-mu_a * norm(pi - pt) * 20.f / costheta(wt));

	Spectrum TT = select(active_tt, T1 * A_t * T2, 0.f);

	// sample TRT lobe
	Float phi_tr = dir_phi(wtr);
	Float gamma_mtr = gamma_mi - 2.f * (to_phi(phi_t, a, b) - to_phi(phi_tr, a, b)) + Pi;
	Normal3f wmtr = sphg_dir(-m_tilt, gamma_mtr, a, b);
	Normal3f wmtr_ = sphg_dir(0, gamma_mtr, a, b);
	auto [wh3, pdfh3] = sample_wh(wtr, wmtr, distr, sample_h3);

	/* fresnel coefficient */
        auto [R3, cos_theta_t3, eta_it3, eta_ti3] = fresnel(dot(wtr, wh3), Float(m_inv_eta));
	Vector3f wtrt = refract(wtr, wh3, cos_theta_t3, eta_ti3);
	active_trt &= (cos_theta_t3 != 0); // total internal reflection
	active_trt &= (dot(wtr, wh3) > 0 && dot(wmtr, wtr) > 0 && dot(wtrt, wmtr) < 0 && pdfh3 > 0
		       && G_(wtr, -wtrt, Normal3f(wmtr.x(), 0.f, wmtr.z()), wh3) > 0);
	Spectrum T3 = 1.f - R3;
	Point2f ptr = to_point(gamma_mtr + Pi, a, b);
	/* divide by 0.05 (* 20) because the absorption is defined for hair width 0.1 */
	Spectrum A_tr = exp(-mu_a * norm(pt - ptr) * 20.f / costheta(wtr));
	Spectrum TRT = select(active_trt, T1 * R2 * T3 * A_t * A_tr, 0.f);

	// select lobe based on energy
	Float r = hmean(R);
	Float tt = hmean(TT);
	Float trt = hmean(TRT);
	Float total_energy = r + tt + trt;

	active &= (total_energy > 0 && enoki::isfinite(total_energy));

	sample_lobe *= total_energy;
	Mask selected_r = sample_lobe < r && active_r;
	Mask selected_tt = sample_lobe >= r && sample_lobe < (r + tt) && active_tt;
	Mask selected_trt = sample_lobe >= (r + tt) && active_trt;

        bs.wo = select(selected_r, wr, select(selected_tt, wtt, wtrt));
        bs.eta = 1.f;
	bs.sampled_component = 0;
	bs.sampled_type = +BSDFFlags::GlossyReflection;

        UnpolarizedSpectrum weight =
	    select(selected_r, R / r * total_energy,
		   select(selected_tt, TT / tt * total_energy,
			  select(selected_trt, TRT / trt * total_energy, 0.f)));

	Float visibility = select(selected_r, smith_g1(wr, wmi, wh1) * G_(si.wi, wr, wmi_, wh1),
				  select(selected_tt, smith_g1(-wt, wmi, wh1) * smith_g1(-wtt, wmt, wh2)
					 * G_(si.wi, -wt, wmi_, wh1) * G_(-wt, -wtt, wmt_, wh2),
					 select(selected_trt,
						smith_g1(-wt, wmi, wh1) * smith_g1(-wtr, wmt, wh2) * smith_g1(-wtrt, wmtr, wh3)
						* G_(si.wi, -wt, wmi_, wh1) * G_(-wt, -wtr, wmt_, wh2) * G_(wtr, -wtrt, wmtr_, wh3),
						0.f)));

	// {
	// Float dwh_dwo = select(selected_r, rcp(4.f * dot(wr, wh1)),
	// 		       select(selected_tt,
	// 			      sqr(m_inv_eta) * rcp(squared_norm(-wt + m_inv_eta * wtt)) * dot(-wtt, wh2),
	// 			      select(selected_trt,
	// 				     sqr(m_inv_eta) * rcp(squared_norm(wtr + m_inv_eta * wtrt)) * dot(-wtrt, wh3),
	// 				     0.f)));

	// bs.pdf = abs(dwh_dwo) *
	//     select(selected_r, r / total_energy * pdfh1,
	// 	   select(selected_tt, tt / total_energy * pdfh2,
	// 		  select(selected_trt, trt / total_energy * pdfh3, 0.f)));
	// }

	weight *= visibility;

	/* correction of the cosine foreshortening term */
	// weight *= dot(si.wi, wmi) / dot(si.wi, wmi_);

	/* ensure the same pdf is returned for BSDF and emitter sampling */
	bs.pdf = this->pdf(ctx, si, bs.wo, active);

	return { bs, select(active, weight, 0.f) };
    }

    /* evaluate the R lobe */
    Spectrum eval_r(const SurfaceInteraction3f &si, const Vector3f &wo_) const {
	auto [a, b, e2] = get_abe2(si);

	// in mitsuba we trace ray from the camera
	Vector3f wo = si.wi;
	Vector3f wi = wo_;
	Float phi_i = dir_phi(wi);
	Float phi_o = dir_phi(wo);

	Vector3f wh = normalize(wi + wo);

	// compute valid phi_mi
	/* dot(wi, wmi) > 0 */
	Float phi_m_max1 = acos(max(-m_tan_tilt * tantheta(wi), 0)) + phi_i;
	if (enoki::isnan(phi_m_max1))
	    return 0.f;
	Float phi_m_min1 = -phi_m_max1 + 2.f * phi_i;

	/* dot(wo, wmi) > 0 */
	Float phi_m_max2 = acos(max(-m_tan_tilt * tantheta(wo), 0)) + phi_o;
	if (enoki::isnan(phi_m_max2))
	    return 0.f;
	Float phi_m_min2 = -phi_m_max2 + 2.f * phi_o;

	/* try to wrap range */
	if ((phi_m_max2 - phi_m_min1) > TwoPi) {
	    phi_m_min2 -= TwoPi; 	    phi_m_max2 -= TwoPi;
	}
	if ((phi_m_max1 - phi_m_min2) > TwoPi) {
	    phi_m_min1 -= TwoPi; 	    phi_m_max1 -= TwoPi;
	}

	Float phi_m_min = max(phi_m_min1, phi_m_min2) + 0.001f;
	Float phi_m_max = min(phi_m_max1, phi_m_max2) - 0.001f;

	if (phi_m_min > phi_m_max)
	    return 0.f;

	Float gamma_m_min = to_gamma(phi_m_min, a, b);
	Float gamma_m_max = to_gamma(phi_m_max, a, b);

	if (gamma_m_max < gamma_m_min)
	    gamma_m_max += TwoPi;

	Float integral = 0.f;

	/* initial sample resolution */
	Float res = m_roughness * .7f;
	Float scale = (gamma_m_max - gamma_m_min) * .5f;
	size_t intervals = 2 * ceil(scale/res);
	/* modified resolution based on integral domain */
	res = (gamma_m_max - gamma_m_min) / Float(intervals);
	// integrate using Simpson's rule
	for (size_t i = 0; i <= intervals; i++) {
	    Float gamma_m = gamma_m_min + i * res;
	    Vector3f wm = sphg_dir(m_tilt, gamma_m, a, b);
	    Float weight = (i == 0 || i == intervals)? 0.5f: (i%2 + 1);
	    Float arc_length = sqrt(1.f - e2 * sqr(sin(gamma_m)));
	    integral += weight * D(wm, wh) * G(wi, wo, wm, wh) * arc_length
		* G_(wi, wo, Normal3f(wm.x(), 0.f, wm.z()), wh);
	}
	integral *= (2.f / 3.f * res);

	Float F = std::get<0>(fresnel(dot(wi, wh), Float(m_eta)));
	Float d_o_inv = rsqrt(1.f - e2 * sqr(sin(phi_o)));
	UnpolarizedSpectrum R = 0.125f * F * integral * d_o_inv;

	return R;
    }

    /* evaluate TT + TRT lobe */
    Spectrum eval_tt_trt(const SurfaceInteraction3f & si,
			 const Vector3f& wo_) const {
	Vector3f wo = si.wi;
	Vector3f wi = wo_;
	Float phi_i = dir_phi(wi);
	Float phi_o = dir_phi(wo);

	/* dot(wi, wmi) > 0 */
	Float phi_m_max = acos(max(-m_tan_tilt * tantheta(wi), 0)) + phi_i;
	if (enoki::isnan(phi_m_max))
	    return 0.f;
	Float phi_m_min = -phi_m_max + 2.f * phi_i;

	/* dot(wo, wmo) < 0 */
	Float tmp1 = acos(min(m_tan_tilt * tantheta(wo), 0.f)); //x
	if (enoki::isnan(tmp1))
	    return 0.f;

	// get geometry data from surface interaction
	auto [a, b, e2] = get_abe2(si);

	ScalarFloat res = m_roughness * .8f;

	/* absorption */
	Spectrum wavelengths = get_spectrum(si);
	Spectrum mu_a = fmadd(m_pheomelanin, pheomelanin(wavelengths),
			      m_eumelanin * eumelanin(wavelengths));

	/* Construct a microfacet distribution matching the
	   roughness values at the current surface position.
	*/
	MicrofacetDistribution distr(m_type, m_roughness, true);
	if (m_type == MicrofacetType::Beckmann) {
	    /* sample_visible = true would be too slow for beckmann */
	    distr = MicrofacetDistribution(m_type, m_roughness, false);
	}

	Float gamma_m_min = to_gamma(phi_m_min, a, b);
	Float gamma_m_max = to_gamma(phi_m_max, a, b);
	if (gamma_m_max < gamma_m_min)
	    gamma_m_max += TwoPi;

	Float scale = (gamma_m_max - gamma_m_min) * .5f;
	size_t intervals = 2 * ceil(scale/res);
	res = (gamma_m_max - gamma_m_min)/intervals;
	UnpolarizedSpectrum S_tt = 0.f, S_trt = 0.f;
	for (size_t i = 0; i <= intervals; i++) {
	    Float gamma_mi = gamma_m_min + i * res;
	    Normal3f wmi = sphg_dir(m_tilt, gamma_mi, a, b);
	    Normal3f wmi_ = sphg_dir(0.f, gamma_mi, a, b);

	    /* sample wh1 */
	    Point2f sample1 = const_cast<Sampler&>(*m_sampler).next_2d(true);
	    Normal3f wh1 = std::get<0>(sample_wh(wi, wmi, distr, sample1));

	    Float cos_ih1 = dot(wi, wh1);
	    if (!(cos_ih1 > 1e-5f))
		continue;

	    /* fresnel coefficient */
	    auto [R1, cos_theta_t1, eta_it1, eta_ti1] = fresnel(dot(wi, wh1), Float(m_eta));
	    Float T1 = 1.f - R1;

	    /* refraction at the first interface */
	    Vector3f wt = refract(wi, wh1, cos_theta_t1, eta_ti1);
	    Float phi_t = dir_phi(wt);
	    Float gamma_mt = 2.f * to_phi(phi_t, a, b) - gamma_mi;
	    Vector3f wmt = sphg_dir(-m_tilt, gamma_mt, a, b);
	    Vector3f wmt_ = sphg_dir(0, gamma_mt, a, b);

	    /* Simpson's rule weight */
	    Float weight = (i == 0 || i == intervals)? 0.5f: (i%2 + 1);

	    Normal3f wh2;
	    Point2f pi = to_point(gamma_mi, a, b);
	    Point2f pt = to_point(gamma_mt + Pi, a, b);
	    /* divide by 0.05 (* 20) because the absorption is defined for hair width 0.1 */
	    Spectrum A_t = exp(-mu_a * norm(pi - pt) * 20.f / costheta(wt));
	    Float G1 = G(wi, -wt, wmi, wh1);
	    if (G1 == 0 || G_(wi, -wt, wmi_, wh1) == 0)
		continue;

	    if (dot(wo, wt) < m_inv_eta - 1e-5f) /* total internal reflection */
		goto TRT;

	    wh2 = -wt + m_inv_eta * wo;
	    if (dot(wmt, wh2) < 0) /* microfacet invisible from macronormal */
		goto TRT;

	    {
		Float rcp_norm_wh2 = rcp(norm(wh2));
		wh2 = wh2 * rcp_norm_wh2;

		Float dot_wt_wh2 = dot(-wt, wh2);
		Float T2 = 1.f - std::get<0>(fresnel(dot_wt_wh2, Float(m_inv_eta)));
		Float D2 = D(wh2, wmt) * G(-wt, -wo, wmt, wh2);
		Float arc_length = sqrt(1.f - e2 * sqr(sin(gamma_mt)));
		/* integrand_of_S_tt / pdf_of_sampling_wt */
		Spectrum result = T1 * T2 * D2 * A_t * dot_wt_wh2 * dot(wo, wh2)
		    * sqr(rcp_norm_wh2) * rcp(dot(wt, wmi)) * weight *
		    select(distr.sample_visible(), smith_g1(-wt, wmi, wh1) * dot(wi, wmi),
			   G1 * cos_ih1 / dot(wh1, wmi));
		masked(result, !enoki::isfinite(result)) = 0;
		S_tt += result * arc_length;

	    }

	TRT:
	    Point2f sample2 = const_cast<Sampler&>(*m_sampler).next_2d(true);
	    wh2 = std::get<0>(sample_wh(-wt, wmt, distr, sample2));

	    Float cos_th2 = dot(-wt, wh2);
	    if (!(cos_th2 > 1e-5f))
		continue;

	    /* fresnel coefficient */
	    auto [R2, cos_theta_t2, eta_it2, eta_ti2] = fresnel(cos_th2, Float(m_inv_eta));
	    Vector3f wtr = reflect(wt, wh2);

	    Float G2 = G(-wt, -wtr, wmt, wh2);
	    if (G2 == 0 || G_(-wt, -wtr, wmt_, wh2) == 0)
		continue;

	    if (dot(-wtr, wo) < m_inv_eta - 1e-5f) /* total internal reflection */
		continue;

	    Float phi_tr = dir_phi(wtr);
	    Float gamma_mtr = gamma_mi - 2.f * (to_phi(phi_t, a, b) - to_phi(phi_tr, a, b)) + Pi;
	    Normal3f wmtr = sphg_dir(-m_tilt, gamma_mtr, a, b);
	    Normal3f wmtr_ = sphg_dir(0, gamma_mtr, a, b);

	    Normal3f wh3 = wtr + m_inv_eta * wo;
	    Float G3 = G(wtr, -wo, wmtr, wh3);
	    if (dot(wmtr, wh3) < 0 || G3 == 0 || G_(wtr, -wo, wmtr_, wh3) == 0)
		continue;

	    Float rcp_norm_wh3 = rcp(norm(wh3));
	    wh3 *= rcp_norm_wh3;

	    Float cos_trh3 = dot(wh3, wtr);
	    Float T3 = 1.f - std::get<0>(fresnel(cos_trh3, Float(m_inv_eta)));

	    Float D3 = D(wh3, wmtr) * G3;
	    Point2f ptr = to_point(gamma_mtr + Pi, a, b);
	     /* divide by 0.05 (or multiply by 20) because the absorption is defined for hair width 0.1 */
	    Spectrum A_tr = exp(-mu_a * norm(pt - ptr) * 20.f / costheta(wtr));

	    Spectrum result = T1 * R2 * T3 * D3 * cos_trh3 * dot(wh3, wo) * sqr(rcp_norm_wh3) *
		A_t * A_tr * weight / (dot(wt, wmi) * dot(wtr, wmt)) *
		select(distr.sample_visible(),
		       smith_g1(-wt, wmi, wh1) * smith_g1(-wtr, wmt, wh2) * dot(wi, wmi) * dot(wt, wmt),
		       -G1 * G2 * cos_ih1 * cos_th2 / (dot(wh1, wmi) * dot(wh2, wmt)));

	    masked(result, !enoki::isfinite(result)) = 0;
	    Float arc_length = sqrt(1.f - e2 * sqr(sin(gamma_mtr)));
	    S_trt += result * arc_length;
	}
	Float d_o_inv = rsqrt(1.f - e2 * sqr(sin(phi_o)));
	return (S_tt + S_trt) * 1.f / 3.f * res * sqr(m_inv_eta) * d_o_inv;
    }

    // evaluate bsdf
    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
		  const Vector3f &wo, Mask active) const override {
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

	UnpolarizedSpectrum R = eval_r(si, wo) + eval_tt_trt(si, wo);

	return select(active, R * rcp(cos(dir_theta(si.wi))), 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
	      const Vector3f &wo, Mask active) const override {
	// Not implemented because it is too expensive to compute
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
	return 1.f;
    }


    void traverse(TraversalCallback *callback) override {
	Base::traverse(callback);
    }

    std::string to_string() const override {
	std::ostringstream oss;
	oss << "RoughHair[" << std::endl
            << "  distribution = " << m_type << "," << std::endl
	    << "  roughness = " << string::indent(m_roughness) << "," << std::endl
	    << "  eta = " << string::indent(m_eta) << "," << std::endl
	    << "  scale tilt = " << string::indent(-m_tilt) << std::endl
	    << "]";
	return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Relative refractive index and its inverse
    ScalarFloat m_eta, m_inv_eta;
    /// Roughness values
    ScalarFloat m_roughness, m_roughness_squared;
    /// Hair scale and its tangent
    ScalarFloat m_tilt, m_tan_tilt;
    /// Sampler for evaluation
    ref<Sampler> m_sampler;
    /// Hair color
    ScalarFloat m_eumelanin;
    ScalarFloat m_pheomelanin;
    bool m_sample_visible;
};

MTS_IMPLEMENT_CLASS_VARIANT(RoughHair, BSDF)
MTS_EXPORT_PLUGIN(RoughHair, "Macrofacet-based hair")
NAMESPACE_END(mitsuba)
