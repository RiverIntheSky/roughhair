#pragma once

#include "hairsegment.h"
#include <mitsuba/render/shape.h>
#include <mitsuba/render/kdtree.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/mstream.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/sampler.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif
#if defined(MTS_ENABLE_OPTIX)
    #include <mitsuba/render/optix/shapes.h>
#endif

NAMESPACE_BEGIN(mitsuba)

/**!
Hair geometry from Cem Yuksel
http://www.cemyuksel.com/research/hairmodels/
 */

#define _CY_HAIR_FILE_SEGMENTS_BIT		1
#define _CY_HAIR_FILE_POINTS_BIT		2
#define _CY_HAIR_FILE_THICKNESS_BIT		4
#define _CY_HAIR_FILE_TRANSPARENCY_BIT	        8
#define _CY_HAIR_FILE_COLORS_BIT		16

#define _CY_HAIR_FILE_INFO_SIZE			88

template <typename Float, typename Spectrum>
class Hair final: public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_id)
    MTS_IMPORT_TYPES(ShapeKDTree)

    static constexpr auto Pi        = math::Pi<ScalarFloat>;
    static constexpr auto InvPi     = math::InvPi<ScalarFloat>;

    using typename Base::ScalarSize;

    Hair(const Properties &props);
    ~Hair();

    // =========================================================================
    //! @{ \name Header
    // =========================================================================

    struct Header
    {
	char			signature[4];	//!< This should be "HAIR"
	unsigned int     	hair_count;	//!< number of hair strands
	unsigned int	        point_count;	//!< total number of points of all strands
	unsigned int    	arrays;		//!< bit array of data in the file

	unsigned int	        d_segments;	//!< default number of segments of each strand
	float			d_thickness;	//!< default thickness of hair strands
	float			d_transparency;	//!< default transparency of hair strands
	float			d_color[3];	//!< default color of hair strands

	char			info[_CY_HAIR_FILE_INFO_SIZE];	//!< information about the file
    };

    // =============================================================
    //! @{ \name Implementation of the \ref Shape interface
    // =============================================================
    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override;

    Mask ray_test(const Ray3f &ray_, Mask active) const override;
    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override;

    ScalarBoundingBox3f bbox() const override { return m_bbox; }

    ScalarFloat solve_cubic(ScalarFloat a, ScalarFloat b, ScalarFloat c, ScalarFloat d) const;

    std::pair<ScalarPoint3f, ScalarFloat> compute_bt(ScalarPoint3f p0, ScalarPoint3f p1, ScalarPoint3f p2) const;

    ScalarPoint3f get_bezier_point(ScalarPoint3f p0, ScalarPoint3f b1, ScalarPoint3f p2, ScalarFloat t) const {
	return (p0 - 2.f * b1 + p2) * sqr(t) + 2.f * (b1 - p0) * t + p0;
    }

    /* F(theta), F'(theta), F''(theta) */
    std::tuple<ScalarPoint3f, ScalarPoint3f, ScalarPoint3f>
    get_bezier123(ScalarFloat theta, ScalarPoint3f p0, ScalarPoint3f b1, ScalarPoint3f p2,
		  ScalarFloat s1, ScalarFloat s2) const {
	ScalarFloat t = theta * (s1 * theta + s2);
	ScalarPoint3f a2 = p0 - 2.f * b1 + p2;
	ScalarPoint3f a1 = 2.f * (b1 - p0);
	ScalarFloat dtdtheta = 2.f * theta * s1 + s2;
	ScalarFloat d2tdtheta2 = 2.f * s1;
	ScalarPoint3f dFdt = 2.f * a2 * t + a1;
	ScalarPoint3f d2Fdt2 = 2.f * a2;
	return {a2 * sqr(t) + a1 * t + p0,
	    dFdt * dtdtheta,
	    dFdt * d2tdtheta2 + d2Fdt2 * sqr(dtdtheta)};
    }

    ScalarFloat surface_area() const override;

    ScalarSize primitive_count() const override {return 1; }

    MTS_INLINE ScalarSize effective_primitive_count() const override;

    void initialize() {
	m_segments = NULL;
	m_points_before_transform = NULL;
	m_transparency = NULL;
	m_colors = NULL;
	m_points = NULL;
    }

    void cleanup() {
	if (m_segments) { delete [] m_segments;}
	if (m_points_before_transform) { delete [] m_points_before_transform; delete [] m_points;}
	if (m_transparency) { delete [] m_transparency;}
	if (m_colors) { delete [] m_colors;}
    }

    //! @}
    // =============================================================

    /// Return a human-readable representation
    std::string to_string() const override;

    MTS_DECLARE_CLASS()
private:
    ScalarBoundingBox3f m_bbox;
    ref<ShapeKDTree> m_kdtree;
    //! Hair file header
    Header m_header;
    unsigned short* m_segments;
    float* m_points_before_transform;
    float* m_transparency;
    float* m_colors;
    ScalarPoint3f* m_points;
    ScalarFloat m_a, m_b;
};

NAMESPACE_END(mitsuba)
