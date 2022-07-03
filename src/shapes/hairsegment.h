#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-hairsegment:

HairSegment (:monosp:`hairsegment`)
----------------------------------------------------

.. pluginparameters::


 * - e0
   - |point|
   - Object-space starting point of the hair segment's centerline.
     (Default: (0, 0, 0))
 * - e1
   - |point|
   - Object-space endpoint of the hairsegment's centerline (Default: (0, 0, 1))
 * - radius
   - |float|
   - Radius of the hair segment in object-space units (Default: 0.05)
 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
     not permitted! (Default: none, i.e. object space = world space)

This shape plugin describes a simple hair segment intersection primitive,
the underlying shape is a ray-facing stripe.
It is part of the hair primitive.

A simple example for instantiating a HairSegment:

.. code-block:: xml

    <shape type="hairsegment">
        <point name="e0" value="0,0,0"/>
        <point name="e1" value="0,0,0.25"/>
        <float name="radius" value="0.05"/>
        <bsdf type="roughhair">
            <string name="distribution" value="beckmann"/>
            <float name="tilt" value="-3"/>
            <float name="eumelanin" value="0.3"/>
            <float name="pheomelanin" value="0.4"/>
            <float name="roughness" value="0.15"/>
        </bsdf>
    </shape>
 */

template <typename Float, typename Spectrum>
class HairSegment final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    HairSegment(const Properties &props) : Base(props) {
        // Update the to_world transform if face points and radius are also provided
        float radius = props.float_("radius", 0.05f);
        ScalarPoint3f e0 = props.point3f("e0", ScalarPoint3f(0.f, 0.f, 0.f)),
                      e1 = props.point3f("e1", ScalarPoint3f(0.f, 0.f, 1.f));

        m_to_world = m_to_world * ScalarTransform4f::translate(e0) *
                                  ScalarTransform4f::to_frame(ScalarFrame3f(e1 - e0)) *
                                  ScalarTransform4f::scale(ScalarVector3f(radius, radius, 1.f));

        update();
        set_children();
    }

    HairSegment(ScalarPoint3f e0, ScalarPoint3f e1,
	       float radius, const Properties &props) : Base(props) {
	Vector3f n = e1 - e0;
	ScalarFloat norm_n = norm(n);

        // Update the to_world transform if face points and radius are also provided
	m_to_world = ScalarTransform4f::translate(e0) *
	    ScalarTransform4f::to_frame(ScalarFrame3f(n/norm_n)) *
	    ScalarTransform4f::scale(ScalarVector3f(radius, radius, norm_n));

	update();
	set_children();
    }

    void update() {
	// Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-5f || abs(S[0][2]) > 1e-5f || abs(S[1][0]) > 1e-5f ||
            abs(S[1][2]) > 1e-5f || abs(S[2][0]) > 1e-5f || abs(S[2][1]) > 1e-5f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");


        if (!(abs(S[0][0] - S[1][1]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling along the X and Y axes!");

        m_radius = S[0][0];
        m_length = S[2][2];

        if (m_radius <= 0.f) {
            m_radius = std::abs(m_radius);
        }

        // Reconstruct the to_world transform with uniform scaling and no shear
        m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
        m_to_object = m_to_world.inverse();

        m_inv_surface_area = rcp(surface_area());
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarVector3f x1 = m_to_world * ScalarVector3f(m_radius, 0.f, 0.f),
                       x2 = m_to_world * ScalarVector3f(0.f, m_radius, 0.f),
                       x  = sqrt(sqr(x1) + sqr(x2));

        ScalarPoint3f e0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
                      e1 = m_to_world * ScalarPoint3f(0.f, 0.f, m_length);

        /* To bound the HairSegment, it is sufficient to find the
           smallest box containing the two circles at the endpoints. */
        return ScalarBoundingBox3f(min(e0 - x, e1 - x), max(e0 + x, e1 + x));
    }

    ScalarFloat surface_area() const override {
        return 2.f * math::Pi<ScalarFloat> * m_radius * m_length;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

        Ray3f ray = m_to_object.transform_affine(ray_);
        Double mint = Double(ray.mint),
               maxt = Double(ray.maxt);

        Double ox = Double(ray.o.x()),
               oy = Double(ray.o.y()),
               oz = Double(ray.o.z()),
               dx = Double(ray.d.x()),
               dy = Double(ray.d.y()),
               dz = Double(ray.d.z());


        scalar_t<Double> radius = scalar_t<Double>(m_radius),
                         length = scalar_t<Double>(m_length);

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
	Double tmp = dx * ox + dy * oy;
	Double t = -tmp * rcp(sqr(dx) + sqr(dy));
	Double dis_sqr = sqr(ox) + sqr(oy) + tmp * t;

	// todo: where does z intersect?
	Double z_pos = oz + dz * t;
	active &= ((dis_sqr < sqr(radius)) && (z_pos >= 0) && (z_pos <= length));

	// Hair doesn't intersect with the segment on the ray
        Mask out_bounds = !(t <= maxt && t >= mint); // NaN-aware conditionals

	// Hair fully contains the segment of the ray
        Mask in_bounds = t < mint && t > maxt;

	// Ray origin inside the hair
	Mask origin_inside = sqr(ox) + sqr(oy) <= sqr(radius);

	pi.t = select(active && !in_bounds && !out_bounds && !origin_inside, t, math::Infinity<Float>);

        pi.shape = this;

	// std::cout << printn(ray.o)  printn(ray.d) printn(pi.t)  std::endl;

        return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

        Ray3f ray = m_to_object.transform_affine(ray_);

        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double ox = Double(ray.o.x()),
               oy = Double(ray.o.y()),
               oz = Double(ray.o.z()),
               dx = Double(ray.d.x()),
               dy = Double(ray.d.y()),
               dz = Double(ray.d.z());

	scalar_t<Double> radius = scalar_t<Double>(m_radius),
	                 length = scalar_t<Double>(m_length);

	Mask origin_inside = sqr(ox) + sqr(oy) < sqr(radius);

	Double tmp = dx * ox + dy * oy;
	Double t = -tmp * rcp(sqr(dx) + sqr(dy));
	Double dis_sqr = sqr(ox) + sqr(oy) + tmp * t;

	// todo: where does z intersect?
	Double z_pos = oz + dz * t;

        active &= ((dis_sqr < sqr(radius)) && (z_pos >= 0) && (z_pos <= length));

	// Hair doesn't intersect with the segment on the ray
        Mask out_bounds = !(t <= maxt && t >= mint); // NaN-aware conditionals

	// Hair fully contains the segment of the ray
        Mask in_bounds = t < mint && t > maxt;

	Mask valid_intersection =
	    active && !origin_inside && !out_bounds && !in_bounds;
	// myprint(valid_intersection);

        return valid_intersection;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);

        si.p = ray(pi.t);

        Vector3f local = m_to_object.transform_affine(si.p);

        Float phi = atan2(local.y(), local.x());
        masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

        si.uv = Point2f(phi * math::InvTwoPi<Float>, local.z() / m_length);

        Vector3f dp_dv = Vector3f(0.f, 0.f, m_length);
        si.dp_dv = m_to_world.transform_affine(dp_dv);
	si.dp_du = cross(ray.d, si.dp_dv);
        si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

        /* Mitigate roundoff error issues by a normal shift of the computed
           intersection point */
        // si.p += si.n * (m_radius - norm(head<2>(local)));

        si.sh_frame.n = si.n;
        si.time = ray.time;

        if (has_flag(flags, HitComputeFlags::dNSdUV)) {
            si.dn_du = si.dp_du / m_radius;
            si.dn_dv = Vector3f(0.f);
        }

	// store h in si.dn_du.x()
	Ray3f ray_ = m_to_object.transform_affine(ray);

	using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

        Double ox = Double(ray_.o.x()),
               oy = Double(ray_.o.y()),
               dx = Double(ray_.d.x()),
 	       dy = Double(ray_.d.y());

	scalar_t<Double> radius = scalar_t<Double>(m_radius);

        si.dn_du.x() = (dy*ox-dx*oy)*rsqrt(sqr(dy)+sqr(dx)) / radius;

        return si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HairSegment[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  length = "  << m_length << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFloat m_radius, m_length;
    ScalarFloat m_inv_surface_area;
};

MTS_IMPLEMENT_CLASS_VARIANT(HairSegment, Shape)

NAMESPACE_END(mitsuba)
