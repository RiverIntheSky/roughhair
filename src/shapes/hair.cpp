#include "hair.h"
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <fstream>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <unordered_map>
#include <unordered_set>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-hair:

Hair (:monosp:`hair`)
----------------------------------------------------

.. pluginparameters::


 * - filename
   - |string|
   - The .hair file that stores the hair geometry
 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
   not permitted! (Default: none, i.e. object space = world space)
 * - a
   - |float|
   - semi-major axis
 * - b
   - |float|
   - semi-minor axis

   This shape plugin describes the Cem Yuksel hair geometry retrieved
   from http://www.cemyuksel.com/research/hairmodels/, with elliptical hairsegment primitives,
   and subdivided segments computed via Bézier interpolations splines
   (http://www.cemyuksel.com/research/interpolating_splines).
   The semi-minor axis of the ellipse is aligned with the curvature vector
   A simple example for instantiating a hair shape:

   .. code-block:: xml

    <shape type="hair">
    <transform name="to_world">
        <rotate x="1" angle="-90"/>
        <translate y="-5"/>
        <rotate y="1" angle="-90"/>
    </transform>
    <float name="a" value="0.05"/>
    <float name="b" value="0.03"/>
    <string name="filename" value="meshes/wCurly.hair"/>
    <bsdf type="roughhair">
        <string name="distribution" value="ggx"/>
        <float name="tilt" value="-3"/>
        <float name="eumelanin" value="0.02"/>
        <float name="pheomelanin" value="1.5"/>
        <float name="roughness" value="0.1"/>
    </bsdf>
  </shape>
*/


MTS_VARIANT Hair<Float, Spectrum>::Hair(const Properties &props):
    Shape<Float, Spectrum>(props) {
    m_kdtree = new ShapeKDTree(props);

     // Extract scale
    auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

    auto fs = Thread::thread()->file_resolver();
    fs::path file_path = fs->resolve(props.string("filename"));

    std::string file_name = file_path.filename().string();

    auto fail = [&](const char *descr) {
	Throw("Error while loading file \"%s\": %s!", file_name, descr);
    };

    Log(Info, "Loading hair from \"%s\" ..", file_name);
    if (!fs::exists(file_path))
	fail("file not found");

    ref<Stream> stream = new FileStream(file_path);
    Timer timer;

    stream->read(&m_header, sizeof(Header));

    initialize();

    // Check if this is a hair file
    if (strncmp(m_header.signature, "HAIR", 4) != 0 )
	Throw("Not a hair file");

    // Read segments array
    if (m_header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT ) {
	m_segments = new unsigned short[m_header.hair_count];
	stream->read(m_segments, sizeof(unsigned short) * m_header.hair_count);
    }

    // Read points array
    if (m_header.arrays & _CY_HAIR_FILE_POINTS_BIT) {
	m_points_before_transform = new float[m_header.point_count * 3];
	stream->read(m_points_before_transform, sizeof(float) * m_header.point_count * 3);
    }

    // Read transparency array
    if (m_header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT ) {
	m_transparency = new float[m_header.point_count];
	stream->read(m_transparency, sizeof(float) * m_header.point_count);
    }

    // Read colors array
    if (m_header.arrays & _CY_HAIR_FILE_COLORS_BIT ) {
	m_colors = new float[m_header.point_count * 3];
	stream->read(m_colors, sizeof(float) * m_header.point_count * 3);
    }

    // apply transformation to vertices
    if (m_points_before_transform) {
	m_points = new ScalarPoint3f[m_header.point_count];
	for (size_t i = 0; i < m_header.point_count; i++)
	    m_points[i] = m_to_world * ScalarPoint3f(m_points_before_transform[3*i],
						     m_points_before_transform[3*i+1],
						     m_points_before_transform[3*i+2]);
    }

    // apply transformation to axes
    m_a = props.float_("a", 0.05f) * S[0][0]; /* semi-major axis */
    m_b = props.float_("b", 0.03f) * S[0][0]; /* semi-minor axis */

    size_t p = 0; // point index
    for (size_t hi = 0; hi < m_header.hair_count; hi++){ // iterating hair strands
	size_t num_seg = (m_segments) ? m_segments[hi] : m_header.d_segments;
	if (num_seg > 0) {
	    ScalarPoint3f p0 = m_points[p];
	    ScalarPoint3f p1 = m_points[p + 1];
	    ScalarPoint3f p2 = m_points[p + 2];
	    auto [b1, t1] = compute_bt(p0, p1, p2);
	    ScalarVector3f x;
	    for (size_t si = 0; si < num_seg; si++,p++) {
		ScalarFloat s11 = (0.5f - t1) * 4.f * sqr(InvPi);
		ScalarFloat s12 = InvPi - s11 * Pi;
		if (si > 0 && si < num_seg - 1) { /* interpolation spline */
		    ScalarPoint3f p3 = m_points[p + 2];
		    auto [b2, t2] = compute_bt(p1, p2, p3);

		    ScalarFloat s21 = (0.5f - t2) * 4.f * sqr(InvPi);
		    ScalarFloat s22 = InvPi - s21 * Pi;

		    size_t depth = max(0, floor(log2(norm(p1 - p2) * 6.f))); /* subdivision depth */
		    size_t num_sub_seg = pow(2, (int)depth);
		    ScalarFloat dtheta = 0.5f * Pi / ScalarFloat(num_sub_seg);

		    for (size_t sub_seg = 0; sub_seg < num_sub_seg; sub_seg++) {
			ScalarFloat theta1 = 0.f + sub_seg * dtheta;
			ScalarFloat theta2 = 0.f + (sub_seg + 1.f) * dtheta;
			ScalarFloat thetamid = .5f * (theta1 + theta2);

			ScalarFloat s1_sqr = sqr(sin(theta1));
			ScalarFloat s2_sqr = sqr(sin(theta2));

			ScalarFloat t11 = s11 * sqr(theta1 + 0.5f * Pi) + s12 * (theta1 + 0.5f * Pi);
			ScalarFloat t12 = s11 * sqr(theta2 + 0.5f * Pi) + s12 * (theta2 + 0.5f * Pi);
			ScalarFloat t21 = s21 * sqr(theta1) + s22 * theta1;
			ScalarFloat t22 = s21 * sqr(theta2) + s22 * theta2;

			ScalarPoint3f F11 = get_bezier_point(p0, b1, p2, t11);
			ScalarPoint3f F12 = get_bezier_point(p0, b1, p2, t12);
			ScalarPoint3f F21 = get_bezier_point(p1, b2, p3, t21);
			ScalarPoint3f F22 = get_bezier_point(p1, b2, p3, t22);

			ScalarPoint3f C1 = lerp(F11, F21, s1_sqr);
			ScalarPoint3f C2 = lerp(F12, F22, s2_sqr);

			auto [st, ct] = sincos(thetamid);
			auto ct_sqr = sqr(ct);
			auto st_sqr = 1.f - ct_sqr;
			auto [F1, F1_, F1__] = get_bezier123(thetamid + 0.5f * Pi, p0, b1, p2, s11, s12);
			auto [F2, F2_, F2__] = get_bezier123(thetamid, p1, b2, p3, s21, s22);

			/* first order derivative */
			ScalarVector3f C_ = 2.f * ct * st * (F2 - F1) + ct_sqr * F1_ + st_sqr * F2_;
			/* second order derivative */
			ScalarVector3f C__ = 2.f * (ct_sqr - st_sqr) * (F2 - F1)
			    + 4.f * ct * st * (F2_ - F1_)
			    + ct_sqr * F1__ + st_sqr * F2__;
			/* binormal vector */
			ScalarVector3f x = cross(C_, C__);
			ref<HairSegment<Float, Spectrum>> hair_seg
			    = new HairSegment<Float, Spectrum>(C1, C2, x, m_a, m_b, props);
			m_kdtree->add_shape(hair_seg);
		    }
		    p0 = p1;
		    p1 = p2;
		    p2 = p3;
		    b1 = b2;
		    t1 = t2;
		} else { /* Bézier spline */
		    size_t depth;
		    if (si == 0) {
			depth = max(0, floor(log2(norm(p1 - p0) * 6.f))); /* subdivision depth */
		    } else {
			depth = max(0, floor(log2(norm(p1 - p2) * 6.f))); /* subdivision depth */
		    }
		    size_t num_sub_seg = pow(2, (int)depth);
		    ScalarFloat dtheta = 0.5f * Pi / ScalarFloat(num_sub_seg);

		    for (size_t sub_seg = 0; sub_seg < num_sub_seg; sub_seg++) {
			ScalarFloat theta1 = 0.f + sub_seg * dtheta;
			ScalarFloat theta2 = 0.f + (sub_seg + 1.f) * dtheta;
			ScalarFloat thetamid = .5f * (theta1 + theta2);
			ScalarFloat t11, t12;
			if (si == 0) {
			    t11 = s11 * sqr(theta1) + s12 * (theta1);
			    t12 = s11 * sqr(theta2) + s12 * (theta2);
			} else {
			    t11 = s11 * sqr(theta1 + 0.5f * Pi) + s12 * (theta1 + 0.5f * Pi);
			    t12 = s11 * sqr(theta2 + 0.5f * Pi) + s12 * (theta2 + 0.5f * Pi);
			}

			ScalarPoint3f F11 = get_bezier_point(p0, b1, p2, t11);
			ScalarPoint3f F12 = get_bezier_point(p0, b1, p2, t12);

			auto [F1, F1_, F1__] = get_bezier123(thetamid + 0.5f * Pi, p0, b1, p2, s11, s12);
			ScalarVector3f x = cross(F1_, F1__);

			ref<HairSegment<Float, Spectrum>> hair_seg
			    = new HairSegment<Float, Spectrum>(F11, F12, x, m_a, m_b, props);
			m_kdtree->add_shape(hair_seg);
		    }
		}
	    }
	    p++;
	}
    }

    if (!m_kdtree->ready())
	m_kdtree->build();

    m_bbox = m_kdtree->bbox();

    cleanup();
}

MTS_VARIANT std::pair<typename Hair<Float, Spectrum>::ScalarPoint3f, typename Hair<Float, Spectrum>::ScalarFloat>
Hair<Float, Spectrum>::compute_bt(ScalarPoint3f p0, ScalarPoint3f p1, ScalarPoint3f p2) const {
    ScalarVector3f v0 = p0 - p1;
    ScalarVector3f v2 = p2 - p1;
    ScalarFloat c = dot(v0, v2);
    ScalarFloat t = solve_cubic(-dot(v0, v0), -c/3.f, c/3.f, dot(v2, v2));
    return {(p1 - sqr(1.f - t) * p0 - sqr(t) * p2) / (2.f * (1.f - t) * t), t};
}

MTS_VARIANT typename Hair<Float, Spectrum>::ScalarFloat
Hair<Float, Spectrum>::solve_cubic(ScalarFloat d, ScalarFloat c,
 				   ScalarFloat b, ScalarFloat a) const {
    ScalarFloat value = (d + 3.f * c + 3.f * b + a) * .125f;
    if (value >=  1e-6f) return solve_cubic(d, (d + c) * .5f, (d + 2*c + b) * .25f, value) * .5f;
    if (value <= -1e-6f) return .5f + solve_cubic(value, (c + 2*b + a) * .25f, (b + a) * .5f, a) * .5f;
    return .5f;
}

MTS_VARIANT Hair<Float, Spectrum>::~Hair() {
}

MTS_VARIANT typename Hair<Float, Spectrum>::PreliminaryIntersection3f
Hair<Float, Spectrum>::ray_intersect_preliminary(const Ray3f &ray_,
						    Mask active) const {
    return m_kdtree->template ray_intersect_preliminary<false>(ray_, active);
}

MTS_VARIANT typename Hair<Float, Spectrum>::Mask
Hair<Float, Spectrum>::ray_test(const Ray3f &ray_,
				   Mask active) const {
    return m_kdtree->template ray_intersect_preliminary<true>(ray_, active).is_valid();
}

MTS_VARIANT typename Hair<Float, Spectrum>::SurfaceInteraction3f
Hair<Float, Spectrum>::compute_surface_interaction(const Ray3f &ray,
						      PreliminaryIntersection3f pi,
						      HitComputeFlags flags,
						      Mask active) const {
    MTS_MASK_ARGUMENT(active);
    return pi.shape->compute_surface_interaction(ray, pi, flags, active);
}

MTS_VARIANT typename Hair<Float, Spectrum>::ScalarFloat
Hair<Float, Spectrum>::surface_area() const {
    NotImplementedError("surface_area");
}

MTS_VARIANT typename Hair<Float, Spectrum>::ScalarSize
Hair<Float, Spectrum>::effective_primitive_count() const {
#if !defined(MTS_ENABLE_EMBREE)
    if constexpr (!is_cuda_array_v<Float>)
		     return m_kdtree->primitive_count();
#endif

}

MTS_VARIANT std::string Hair<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "Hair[" << std::endl
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS_VARIANT(Hair, Shape)
MTS_EXPORT_PLUGIN(Hair, "Hair shape");
NAMESPACE_END(mitsuba)
