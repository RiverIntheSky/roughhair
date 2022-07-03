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
the underlying shape is a stripe with orientation-dependent width.
It is part of the hair primitive.

A simple example for instantiating a HairSegment:

.. code-block:: xml

    <shape type="hairsegment">
        <point name="e0" value="0,-0.5,0"/>
        <point name="e1" value="0,0.5,0"/>
        <float name="a" value="0.02"/>
        <float name="b" value="0.01"/>
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
	m_a = props.float_("a", 1.f); /* semi-major axis */
	m_b = props.float_("b", 0.5f); /* semi-minor axis */
        ScalarPoint3f e0 = props.point3f("e0", ScalarPoint3f(0.f, 0.f, 0.f)),
                      e1 = props.point3f("e1", ScalarPoint3f(0.f, 1.f, 0.f));

	ScalarFloat length = norm(e1 - e0);
	ScalarFrame3f rot_((e1 - e0) / length);
	ScalarFrame3f rot(rot_);
	rot.n = -rot_.t;
	rot.t = rot_.n;

        m_to_world = m_to_world * ScalarTransform4f::translate(e0) *
	                          ScalarTransform4f::to_frame(rot) *
	                          ScalarTransform4f::scale(ScalarVector3f(m_a, length, m_a));

        update();
        set_children();
    }

    HairSegment(ScalarPoint3f e0, ScalarPoint3f e1,
		float radius, const Properties &props) : m_a(radius), m_b(radius), Base(props) {
	Vector3f y = e1 - e0;
	ScalarFloat norm_y = norm(y);

	ScalarFrame3f rot_(y/norm_y);
	ScalarFrame3f rot(rot_);
	rot.n = -rot_.t;
	rot.t = rot_.n;

        // Update the to_world transform if face points and radius are also provided
	m_to_world = ScalarTransform4f::translate(e0) *
	             ScalarTransform4f::to_frame(rot) *
	             ScalarTransform4f::scale(ScalarVector3f(radius, norm_y, radius));

	update();
	set_children();
    }

    HairSegment(ScalarPoint3f e0, ScalarPoint3f e1, ScalarVector3f x,
		float a, float b, const Properties &props) : Base(props), m_a(a), m_b(b) {
	Vector3f y = e1 - e0;
	ScalarFloat norm_y = norm(y);

	ScalarFrame3f rot;
	rot.s = normalize(x);
	rot.t = y / norm_y;
	rot.n = normalize(cross(rot.s, rot.t));
	rot.s = normalize(cross(rot.t, rot.n));

        // Update the to_world transform if face points and radius are also provided
	m_to_world = ScalarTransform4f::translate(e0) *
	             ScalarTransform4f::to_frame(rot) *
	             ScalarTransform4f::scale(ScalarVector3f(a, norm_y, a));

	update();
	set_children();
    }

    void update() {
	// Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        // Unstable for small radius
        // if (abs(S[0][1]) > 1e-5f || abs(S[0][2]) > 1e-5f || abs(S[1][0]) > 1e-5f ||
        //     abs(S[1][2]) > 1e-5f || abs(S[2][0]) > 1e-5f || abs(S[2][1]) > 1e-5f)
        //     Log(Warn, "'to_world' transform shouldn't contain any shearing!");


        // if (!(abs(S[0][0] - S[2][2]) < 1e-6f))
        //     Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling along the X and Y axes!");

	Float radius = S[0][0];
	m_b = m_b / m_a * radius;
        m_a = radius;
        m_length = S[1][1];

        // Reconstruct the to_world transform with uniform scaling and no shear
        m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
        m_to_object = m_to_world.inverse();
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarVector3f x1 = m_to_world * ScalarVector3f(m_a, 0.f, 0.f),
	               x2 = m_to_world * ScalarVector3f(0.f, 0.f, m_b),
                       x  = sqrt(sqr(x1) + sqr(x2));

        ScalarPoint3f e0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
                      e1 = m_to_world * ScalarPoint3f(0.f, m_length, 0.f);

        /* To bound the HairSegment, it is sufficient to find the
           smallest box containing the two circles at the endpoints. */
        return ScalarBoundingBox3f(min(e0 - x, e1 - x), max(e0 + x, e1 + x));
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

        scalar_t<Double> a = scalar_t<Double>(m_a),
	                 b = scalar_t<Double>(m_b),
                         length = scalar_t<Double>(m_length);

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
	Double tmp = dx * ox + dz * oz;
	Double t = -tmp * rcp(sqr(dx) + sqr(dz));

	Double x_pos = ox + dx * t,
	       y_pos = oy + dy * t,
	       z_pos = oz + dz * t;
	active &= (sqr(x_pos / a) + sqr(z_pos / b) < 1.0) && (y_pos >= 0) && (y_pos <= length);

	// Hair doesn't intersect with the segment on the ray
	Mask out_bounds = !(t <= maxt && t >= mint); // NaN-aware conditionals

	// Hair fully contains the segment of the ray
	Mask in_bounds = t < mint && t > maxt;

	// Ray origin inside the hair
	Mask origin_inside = sqr(ox / a) + sqr(oz / b) < 1.0;

	pi.t = select(active && !in_bounds && !out_bounds && !origin_inside, t, math::Infinity<Float>);

        pi.shape = this;

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

	scalar_t<Double> a = scalar_t<Double>(m_a),
	                 b = scalar_t<Double>(m_b),
	                 length = scalar_t<Double>(m_length);

	Double tmp = dx * ox + dz * oz;
	Double t = -tmp * rcp(sqr(dx) + sqr(dz));

	Double x_pos = ox + dx * t,
	       y_pos = oy + dy * t,
	       z_pos = oz + dz * t;
	active &= (sqr(x_pos / a) + sqr(z_pos / b) < 1.0) && (y_pos >= 0) && (y_pos <= length);

	// Hair doesn't intersect with the segment on the ray
        Mask out_bounds = !(t <= maxt && t >= mint); // NaN-aware conditionals

	// Hair fully contains the segment of the ray
        Mask in_bounds = t < mint && t > maxt;

	// Ray origin inside the hair
	Mask origin_inside = sqr(ox / a) + sqr(oz / b) < 1.0;

	Mask valid_intersection =
	    active && !origin_inside && !out_bounds && !in_bounds;

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

        Float phi = atan2(local.x(), local.z());
        masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

        si.uv = Point2f(phi * math::InvTwoPi<Float>, local.y() / m_length);

        Vector3f dp_dv = Vector3f(0.f, m_length, 0.f);
        si.dp_dv = m_to_world.transform_affine(dp_dv);
	si.dp_du = m_to_world.transform_affine(Vector3f(1.f, 0.f, 0.f));
        si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

        si.sh_frame.n = si.n;
        si.time = ray.time;

	// store a, b, e^2 in si.dn_du.x()
	// just borrow any vector that we do not need in the rendering
	si.dn_du.x() = m_a;
	si.dn_du.y() = m_b;
	si.dn_du.z() = 1.f - sqr(m_b / m_a); // squared eccentricity

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
	    << "  semi-major axis = "  << m_a << "," << std::endl
	    << "  semi-minor axis = "  << m_b << "," << std::endl
            << "  length = "  << m_length << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFloat m_a, m_b, m_length;
};

MTS_IMPLEMENT_CLASS_VARIANT(HairSegment, Shape)

NAMESPACE_END(mitsuba)
