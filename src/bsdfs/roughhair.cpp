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

	// whether to use analytical integration of Microfacet distribution function
	m_analytical = props.bool_("analytical", false);

	// roughness
        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann") {
                m_type = MicrofacetType::Beckmann;
		if (m_analytical)
		    Log(Warn, "Analytical solution of the R lobe exists only for GGX distribution");
	    }
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
    MTS_INLINE Vector3f sph_dir(Float theta, Float gamma) const {
	auto [sin_theta, cos_theta] = sincos(theta);
	auto [sin_gamma,   cos_gamma]   = sincos(gamma);
	return Vector3f(sin_gamma * cos_theta, sin_theta, cos_gamma * cos_theta);
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

	// generate sample
	Float sample_lobe = sample1;
	Float sample_h = const_cast<Sampler&>(*m_sampler).next_1d(active);
	Point2f sample_h1 = sample2;
	Point2f sample_h2 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	Point2f sample_h3 = const_cast<Sampler&>(*m_sampler).next_2d(active);


        // Float sin_phi_mi = si.dn_du.x();	  /* Use offset h directly from intersection data */
	Float sin_phi_mi = sample_h * 2.f - 1.f;  /* Sample offset h = -sin(phi_m)*/
	Float cos_phi_mi = safe_sqrt(1.f - sqr(sin_phi_mi));
	auto [st, ct] = sincos(m_tilt);
	Normal3f wmi(sin_phi_mi * ct, st, cos_phi_mi * ct); /* mesonormal */
	Normal3f wmi_(sin_phi_mi, 0.f, cos_phi_mi); /* macronormal */

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
	Float phi_mi = atan2f(sin_phi_mi, cos_phi_mi);
	Float phi_mt = 2.f * phi_t - phi_mi;
	Normal3f wmt = sph_dir(-m_tilt, phi_mt);
	Normal3f wmt_ = sph_dir(0, phi_mt);
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
	Float cos_gamma_t = -cos(phi_t - phi_mi);
	Float cos_theta_wt = sqrt(1.f - sqr(wt.y()));
	Spectrum A_t = exp(-mu_a * (2.f * cos_gamma_t / cos_theta_wt));

	Spectrum TT = select(active_tt, T1 * A_t * T2, 0.f);

	// sample TRT lobe
	Float phi_tr = dir_phi(wtr);
	Float phi_mtr = phi_mi - 2.f * (phi_t - phi_tr) + Pi;
	Normal3f wmtr = sph_dir(-m_tilt, phi_mtr);
	Normal3f wmtr_ = sph_dir(0, phi_mtr);
	auto [wh3, pdfh3] = sample_wh(wtr, wmtr, distr, sample_h3);

	/* fresnel coefficient */
        auto [R3, cos_theta_t3, eta_it3, eta_ti3] = fresnel(dot(wtr, wh3), Float(m_inv_eta));
	Vector3f wtrt = refract(wtr, wh3, cos_theta_t3, eta_ti3);
	active_trt &= (cos_theta_t3 != 0); // total internal reflection
	active_trt &= (dot(wtr, wh3) > 0 && dot(wmtr, wtr) > 0 && dot(wtrt, wmtr) < 0 && pdfh3 > 0
		       && G_(wtr, -wtrt, Normal3f(wmtr.x(), 0.f, wmtr.z()), wh3) > 0);
	Spectrum T3 = 1.f - R3;
	Float cos_gamma_t2 = -cos(phi_tr - phi_mt);
	Float cos_theta_wtr = sqrt(1.f - sqr(wtr.y()));
	Spectrum A_tr = exp(-mu_a * (2.f * cos_gamma_t2 / cos_theta_wtr));
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
    Spectrum eval_r(const Vector3f &wi, const Vector3f &wo) const {
	UnpolarizedSpectrum R = 0.f;

	Vector3f wh = normalize(wi + wo);

	Float phi_o = dir_phi(wo);
	Float phi_h = dir_phi(wh);

	// compute valid phi_mi
	/* dot(wi, wmi) > 0 */
	Float phi_m_max1 = acos(max(-m_tan_tilt * tantheta(wi), 0));

	if (enoki::isnan(phi_m_max1))
	    return 0.f;
	Float phi_m_min1 = -phi_m_max1;

	/* dot(wo, wmi) > 0 */
	Float phi_m_max2 = acos(max(-m_tan_tilt * tantheta(wo), 0)) + phi_o;
	if (enoki::isnan(phi_m_max2))
	    return 0.f;
	Float phi_m_min2 = -phi_m_max2 + 2.f * phi_o;

	Float phi_m_min = max(phi_m_min1, phi_m_min2) + 1e-5f;
	Float phi_m_max = min(phi_m_max1, phi_m_max2) - 1e-5f;

	if (phi_m_min > phi_m_max)
	    return 0.f;

	Float integral = 0.f;
	Float d_max = phi_h - phi_m_max;
	Float d_min = phi_h - phi_m_min;
	if (m_analytical && m_type == MicrofacetType::GGX) { // TODO: beckmann
	    if (m_tilt == 0.f) {
		Float A = (m_roughness_squared - 1) * sqr(costheta(wh));
		Float temp1 = A * rcp(A + 1) * (sin(2*d_max) * rcp(A*cos(2*d_max)+A+2.f) - sin(2*d_min) * rcp(A*cos(2*d_min)+A+2.f));
		Float temp2 = (A + 2) * pow(A + 1, Float(-1.5)) * (atan(tan(d_min)*rsqrt(A + 1)) - atan(tan(d_max)*rsqrt(A + 1)));
		integral = temp1 + temp2;
	    } else {
		auto [sm, cm] = sincos(m_tilt);
		Float C = sqrt(1.f - m_roughness_squared);
		Float A = cm * costheta(wh) * C;
		Float B = sm * sintheta(wh) * C;
		Float A2 = sqr(A);
		Float B2 = sqr(B);
		Float tmp1 = rsqrt(sqr(B - 1.f) - A2);
		Float tmp2 = rsqrt(sqr(B + 1.f) - A2);

		auto [smax, cmax] = sincos(d_max);
		auto [smin, cmin] = sincos(d_min);
		Float tmax = smax / (1.f + cmax);
		Float tmin = smin / (1.f + cmin);

		Float temp1 = 2.f * (A2 - B2 + 3.f*B - 2) * sqr(tmp1) * tmp1 *
		    (atan((A - B + 1.f) * tmp1 * tmax) -
		     atan((A - B + 1.f) * tmp1 * tmin));
		Float temp2 = 2.f * (A2 - B2 - 3.f*B - 2) * sqr(tmp2) * tmp2 *
		    (atan((B - A + 1.f) * tmp2 * tmax) -
		     atan((B - A + 1.f) * tmp2 * tmin));
		Float temp3 = A * sqr(tmp1) *
		    (smax / (A * cmax + B - 1.f) -
		     smin / (A * cmin + B - 1.f));
		Float temp4 = A * sqr(tmp2) *
		    (smax / (A * cmax + B + 1.f) -
		     smin / (A * cmin + B + 1.f));

		integral = 0.5f * (temp1 + temp2 + temp3 + temp4);
	    }
	    integral *= m_roughness_squared * math::InvTwoPi<Float>;
	} else { /* falls back to numerical integration */
	    /* initial sample resolution */
	    Float res = m_roughness * .7f;
	    Float scale = (phi_m_max - phi_m_min) * .5f;
	    size_t intervals = 2 * ceil(scale/res) + 1;
	    /* modified resolution based on integral domain */
	    res = (phi_m_max - phi_m_min) / Float(intervals);
	    // integrate using Simpson's rule
	    for (size_t i = 0; i < intervals; i++) {
		Float phi_m = phi_m_min + i * res;
		Vector3f wm = sph_dir(m_tilt, phi_m);
		Float weight = (i == 0 || i == intervals - 1)? 0.5f: (i%2 + 1);
		integral += weight * D(wm, wh) * G(wi, wo, wm, wh) * G_(wi, wo, Normal3f(wm.x(), 0.f, wm.z()), wh);
	    }
	    integral *= (2.f / 3.f * res);
	}

	Float F = std::get<0>(fresnel(dot(wi, wh), Float(m_eta)));
	R = 0.125f * F * max(0.f, integral);

	return R;
    }

    /* evaluate TT + TRT lobe */
    Spectrum eval_tt_trt(const SurfaceInteraction3f & si,
			 const Vector3f& wo) const {
	/* dot(wi, wmi) > 0 */
	Float phi_m_max = acos(max(-m_tan_tilt * tantheta(si.wi), 0));
	if (enoki::isnan(phi_m_max))
	    return 0.f;
	Float phi_m_min = -phi_m_max;

	/* dot(wo, wmo) < 0 */
	Float tmp1 = acos(min(m_tan_tilt * tantheta(wo), 0.f)); //x
	if (enoki::isnan(tmp1))
	    return 0.f;

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

	Float scale = (phi_m_max - phi_m_min) * .5f;
	size_t intervals = 2 * ceil(scale/res) + 1;
	res = (phi_m_max - phi_m_min)/intervals;
	UnpolarizedSpectrum S_tt = 0.f, S_trt = 0.f;
	for (size_t i = 0; i < intervals; i++) {
	    Float phi_mi = phi_m_min + i * res;
	    Normal3f wmi = sph_dir(m_tilt, phi_mi);

	    /* sample wh1 */
	    Point2f sample1 = const_cast<Sampler&>(*m_sampler).next_2d(true);
	    Normal3f wh1 = std::get<0>(sample_wh(si.wi, wmi, distr, sample1));

	    Float cos_ih1 = dot(si.wi, wh1);
	    if (!(cos_ih1 > 1e-5f))
		continue;

	    /* fresnel coefficient */
	    auto [R1, cos_theta_t1, eta_it1, eta_ti1] = fresnel(dot(si.wi, wh1), Float(m_eta));
	    Float T1 = 1.f - R1;

	    /* refraction at the first interface */
	    Vector3f wt = refract(si.wi, wh1, cos_theta_t1, eta_ti1);
	    Float phi_t = dir_phi(wt);
	    Float phi_mt = 2.f * phi_t - phi_mi;
	    Vector3f wmt = sph_dir(-m_tilt, phi_mt);

	    /* Simpson's rule weight */
	    Float weight = (i == 0 || i == intervals - 1)? 0.5f: (i%2 + 1);

	    Normal3f wh2;
	    Spectrum A_t = exp(mu_a * 2.f * cos(phi_t - phi_mi) / costheta(wt));
	    Float G1 = G(si.wi, -wt, wmi, wh1);
	    if (G1 == 0 || G_(si.wi, -wt, Normal3f(wmi.x(), 0.f, wmi.z()), wh1) == 0)
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

		/* integrand_of_S_tt / pdf_of_sampling_wt */
		// Spectrum result;
		// if (distr.sample_visible()) {
		//     result = T1 * T2 * smith_g1(-wt, wmi, wh1) * D2 * A_t
		// 	* dot(si.wi, wmi)
		// 	* dot_wt_wh2 * dot(wo, wh2) * sqr(rcp_norm_wh2)
		// 	* rcp(dot(wt, wmi)) * weight;
		// } else {
		//     result = T1 * T2 * G1 * D2 * A_t * cos_ih1
		// 	* dot_wt_wh2 * dot(wo, wh2) * sqr(rcp_norm_wh2)
		// 	* rcp(dot(wt, wmi)) * weight / dot(wh1, wmi);
		// }
		/* integrand_of_S_tt / pdf_of_sampling_wt */
		Spectrum result = T1 * T2 * D2 * A_t * dot_wt_wh2 * dot(wo, wh2)
		    * sqr(rcp_norm_wh2) * rcp(dot(wt, wmi)) * weight *
		    select(distr.sample_visible(), smith_g1(-wt, wmi, wh1) * dot(si.wi, wmi),
			   G1 * cos_ih1 / dot(wh1, wmi));
		masked(result, !enoki::isfinite(result)) = 0;
		S_tt += result;
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
	    if (G2 == 0 || G_(-wt, -wtr, Normal3f(wmt.x(), 0.f, wmt.z()), wh2) == 0)
		continue;

	    if (dot(-wtr, wo) < m_inv_eta - 1e-5f) /* total internal reflection */
		continue;

	    Float phi_tr = dir_phi(wtr);
	    Float phi_mtr = phi_mi - 2.f * (phi_t - phi_tr) + Pi;
	    Normal3f wmtr = sph_dir(-m_tilt, phi_mtr);

	    Normal3f wh3 = wtr + m_inv_eta * wo;
	    Float G3 = G(wtr, -wo, wmtr, wh3);
	    if (dot(wmtr, wh3) < 0 || G3 == 0 || G_(wtr, -wo, Normal3f(wmtr.x(), 0.f, wmtr.z()), wh3) == 0)
		continue;

	    Float rcp_norm_wh3 = rcp(norm(wh3));
	    wh3 *= rcp_norm_wh3;

	    Float cos_trh3 = dot(wh3, wtr);
	    Float T3 = 1.f - std::get<0>(fresnel(cos_trh3, Float(m_inv_eta)));

	    Float D3 = D(wh3, wmtr) * G3;
	    Spectrum A_tr = exp(mu_a * 2.f * cos(phi_tr - phi_mt) / costheta(wtr));

	    // Spectrum result;
	    // if (distr.sample_visible()) {
	    // 	result = T1 * R2 * T3 * smith_g1(-wt, wmi, wh1) * smith_g1(-wtr, wmt, wh2)
	    // 	    * D3 * cos_trh3 * dot(si.wi, wmi) * dot(wt, wmt)
	    // 	    * dot(wh3, wo)	* sqr(rcp_norm_wh3) * A_t * A_tr * weight
	    // 	    / (dot(wt, wmi) * dot(wtr, wmt));
	    // } else {
	    // 	result = T1 * R2 * T3 * G1 * G2 * D3 * cos_ih1 * cos_th2 * cos_trh3
	    // 	    * dot(wh3, -wo)	* sqr(rcp_norm_wh3) * A_t * A_tr * weight
	    // 	    / (dot(wh1, wmi) * dot(wh2, wmt) * dot(wt, wmi) * dot(wtr, wmt));
	    // }
	    Spectrum result = T1 * R2 * T3 * D3 * cos_trh3 * dot(wh3, wo) * sqr(rcp_norm_wh3) *
		A_t * A_tr * weight / (dot(wt, wmi) * dot(wtr, wmt)) *
		select(distr.sample_visible(),
		       smith_g1(-wt, wmi, wh1) * smith_g1(-wtr, wmt, wh2) * dot(si.wi, wmi) * dot(wt, wmt),
		       -G1 * G2 * cos_ih1 * cos_th2 / (dot(wh1, wmi) * dot(wh2, wmt)));
	    masked(result, !enoki::isfinite(result)) = 0;
	    S_trt += result;
	}
	return (S_tt + S_trt) * 1.f / 3.f * res * sqr(m_inv_eta);
    }

    // evaluate bsdf
    // phi_i == 0
    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
		  const Vector3f &wo, Mask active) const override {
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

	UnpolarizedSpectrum R = eval_r(si.wi, wo) + eval_tt_trt(si,wo);

	return select(active, R * rcp(cos(dir_theta(si.wi))), 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
	      const Vector3f &wo, Mask active) const override {
	MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
	// CAUTIOUS: this is only an estimation of the PDF, is now disabled
	return 1.f;

	// check visibility because of scale tilt
	Float phi_o = dir_phi(wo);
	/* dot(wi, wmi) > 0 */
	Float phi_m_max = acos(max(-m_tan_tilt * tantheta(si.wi), 0));
	if (enoki::isnan(phi_m_max))
	    return 0.f;
	Float phi_m_min = -phi_m_max;

	/* dot(wo, wm) > 0 */
	Float tmp1 = acos(max(-m_tan_tilt * tantheta(wo), 0));
	if (enoki::isnan(tmp1))
	    return 0.f;

	/* absorption */
	Spectrum wavelengths = get_spectrum(si);
	Spectrum mu_a = fmadd(m_pheomelanin, pheomelanin(wavelengths),
					m_eumelanin * eumelanin(wavelengths));

	/* dot(wo, wmi) > 0 */
	Float phi_m_max_r = tmp1 + phi_o;
	Float phi_m_min_r = -tmp1 + phi_o;

	Vector3f wh = normalize(si.wi + wo);

	Float pdf_r(0.f), pdf_tt(0.f), pdf_trt(0.f);

	/* Construct a microfacet distribution matching the
	   roughness values at the current surface position. */
	MicrofacetDistribution distr(m_type, m_roughness, false);

	/* initial sample resolution */
	Float res = m_roughness * .8f;
	Float scale = (phi_m_max - phi_m_min) * .5f;
	size_t intervals = 2 * ceil(scale/res) + 1;
	/* modified resolution based on integral domain */
	res = (phi_m_max - phi_m_min) / Float(intervals);
	// integrate using Simpson's rule
	for (size_t i = 0; i < intervals; i++) {
	    Float phi_mi = phi_m_min + i * res;
	    Normal3f wmi = sph_dir(m_tilt, phi_mi);
	    /* R */
	    /* sample wh1 */
	    Point2f sample1 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	    auto [wh1, pdfh1] = sample_wh(si.wi, wmi, distr, sample1);
	    auto [R1, cos_theta_t1, eta_it1, eta_ti1] = fresnel(dot(si.wi, wh1), Float(m_eta));
	    /* TT */
	    Float T1 = 1.f - R1;
	    Vector3f wt = refract(si.wi, wh1, cos_theta_t1, eta_ti1);
	    Float phi_t = dir_phi(wt);
	    Float phi_mt = 2.f * phi_t - phi_mi;
	    Normal3f wmt = sph_dir(-m_tilt, phi_mt);
	    /* sample wh2 */
	    Point2f sample2 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	    auto [wh2, pdfh2] = sample_wh(-wt, wmt, distr, sample2);
	    Float R2 = std::get<0>(fresnel(dot(wh2, -wt), Float(m_inv_eta)));
	    Spectrum At = exp(mu_a * 2.f * cos(phi_t - phi_mi) / costheta(wt));
	    Float TT = T1 * (1.f - R2) * hmean(At);
	    /* TRT */
	    Vector3f wtr = reflect(wt, wh2);
	    Float phi_tr = dir_phi(wtr);
	    Float twottrpi = -2.f * (phi_t - phi_tr) + Pi;
	    Float phi_mtr = phi_mi + twottrpi;
	    Normal3f wmtr = sph_dir(-m_tilt, phi_mtr);
	    /* sample wh3 */
	    Point2f sample3 = const_cast<Sampler&>(*m_sampler).next_2d(active);
	    auto [wh3, pdfh3] = sample_wh(wtr, wmtr, distr, sample3);
	    Float T3 = 1.f - std::get<0>(fresnel(dot(wh3, wtr), Float(m_inv_eta)));
	    Spectrum Atr = exp(mu_a * 2.f * cos(phi_tr - phi_mt) / costheta(wtr));
	    Float TRT = T1 * R2 * T3 * hmean(At * Atr);

	    Float total_energy = R1 + TT + TRT;

	    if (!(total_energy > 0))
		continue;

	    Float weight = (i == 0 || i == intervals - 1)? 0.5f: (i%2 + 1); /* Simpson */
	    weight *= 0.5f * cos(phi_mi); /* cos(phi)dphi = dh, diameter = 2 */

	    // R
	    pdf_r += select(phi_mi < phi_m_max_r && phi_mi > phi_m_min_r,
			    max(0, R1 / total_energy * weight
				* D(wmi, wh) * smith_g1_visible(si.wi, wmi, wh) * 0.25f),
			    0.f);
	    // TT
	    if (dot(wo, wt) > m_inv_eta) {
		Normal3f wh2 = -wt + m_inv_eta * wo;
		Float rcp_norm_wh2 = rcp(norm(wh2));
		wh2 *= rcp_norm_wh2;
		Float pdf_h2 = D(wmt, wh2) * smith_g1_visible(-wt, wmt, wh2) * dot(-wt, wh2);
		Float dwh2_wtt = sqr(m_inv_eta * rcp_norm_wh2) * dot(-wo, wh2);
		Normal3f wmt_ = sph_dir(0, phi_mt);
		Float result = TT / total_energy * pdf_h2 * dwh2_wtt * weight * G_(-wt, -wo, wmt_, wh2);
		masked(result, !enoki::isfinite(result)) = 0;
		pdf_tt += result;
	    }
            // TRT
	    if (dot(-wtr, wo) > m_inv_eta) {
		Normal3f wmtr_ = sph_dir(0, phi_mtr);
		Normal3f wh3 = wtr + m_inv_eta * wo;
		Float rcp_norm_wh3 = rcp(norm(wh3));
		wh3 *= rcp_norm_wh3;
		Float pdf_h3 = D(wmtr, wh3) * smith_g1_visible(wtr, wmtr, wh3) * dot(wtr, wh3);
		Float dwh3_wtrt = sqr(m_inv_eta * rcp_norm_wh3) * dot(-wo, wh3);
		Float result = TRT / total_energy * pdf_h3 * dwh3_wtrt * weight * G_(wtr, -wo, wmtr_, wh3);
		masked(result, !enoki::isfinite(result)) = 0.f;
		pdf_trt += result;
	    }
	}

	return select(active, (pdf_r + pdf_tt + pdf_trt) * (2.f * res / 3.f), 0.f);
    }


    void traverse(TraversalCallback *callback) override {
	Base::traverse(callback);
    }

    std::string to_string() const override {
	std::ostringstream oss;
	oss << "RoughHair[" << std::endl
            << "  distribution = "   << m_type << "," << std::endl
	    << "  roughness = "      << string::indent(m_roughness) << "," << std::endl
	    << "  scale tilt = "     << string::indent(-m_tilt) << std::endl
	    << "  eumelanin = "      << string::indent(m_eumelanin) << ", "  << std::endl
	    << "  pheomelanin = "    << string::indent(m_pheomelanin) << ", " << std::endl
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
    /// Whether integrate analytically
    bool m_analytical;
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
