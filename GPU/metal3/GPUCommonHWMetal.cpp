// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.


#include "ppsspp_config.h"
#include <algorithm>  // std::remove

#include "Common/Profiler/Profiler.h"

#include "Common/GPU/GraphicsContext.h"
#include "Common/LogReporting.h"
#include "Common/Math/SIMDHeaders.h"
#include "Common/Math/CrossSIMD.h"
#include "Common/Serialize/Serializer.h"
#include "Common/Serialize/SerializeFuncs.h"
#include "Common/Serialize/SerializeList.h"
#include "Common/TimeUtil.h"
#include "GPU/GeDisasm.h"
#include "GPU/GPU.h"
#include "GPU/GPUCommon.h"
#include "GPU/GPUState.h"
#include "Core/Config.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/Debugger/MemBlockInfo.h"
#include "Core/MemMap.h"
#include "Core/Reporting.h"
#include "Core/HLE/HLE.h"
#include "Core/HLE/ErrorCodes.h"
#include "Core/HLE/sceKernelMemory.h"
#include "Core/HLE/sceKernelInterrupt.h"
#include "Core/HLE/sceGe.h"
#include "Core/HLE/scePower.h"
#include "Core/Util/PPGeDraw.h"
#include "Core/MemMapHelpers.h"
#include "GPU/Common/DrawEngineCommon.h"
#include "GPU/metal3/FramebufferManagerMetal.h"
#include "GPU/metal3/TextureCachePolicyMetal.h"
#include "GPU/Common/SoftwareTransformCommon.h"
#include "GPU/Debugger/Debugger.h"
#include "GPU/Debugger/Record.h"
#include "GPU/Debugger/Stepping.h"

bool __KernelIsDispatchEnabled();

#include "Common/Profiler/Profiler.h"

#include "Common/GPU/thin3d.h"
#include "Common/Serialize/Serializer.h"
#include "Common/System/System.h"
#include "Common/Data/Text/StringWriter.h"

#include "Core/System.h"
#include "Core/Config.h"
#include "Core/Util/PPGeDraw.h"

#include "GPU/metal3/GPUCommonHWMetal.h"
#include "GPU/Common/SplineCommon.h"
#include "GPU/Common/DrawEngineCommon.h"
#include "GPU/metal3/TextureCachePolicyMetal.h"
#include "GPU/metal3/FramebufferManagerMetal.h"

struct MetalCommandTableEntry {
	uint8_t cmd;
	uint8_t flags;
	uint64_t dirty;
	GPUCommonHWMetal::CmdFunc func;
};

struct MetalCommandInfo {
	uint64_t flags;
	GPUCommonHWMetal::CmdFunc func;

	// Dirty flags are mashed into the regular flags by a left shift of 8.
	void AddDirty(u64 dirty) {
		flags |= dirty << 8;
	}
	void RemoveDirty(u64 dirty) {
		flags &= ~(dirty << 8);
	}
};

static MetalCommandInfo cmdInfo_[256];

const MetalCommandTableEntry metalCommandTable[] = {
	// From Common. No flushing but definitely need execute.
	{ GE_CMD_OFFSETADDR, FLAG_EXECUTE, 0, &GPUCommon::Execute_OffsetAddr },
	{ GE_CMD_ORIGIN, FLAG_EXECUTE | FLAG_READS_PC, 0, &GPUCommon::Execute_Origin },
	{ GE_CMD_JUMP, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommon::Execute_Jump },
	{ GE_CMD_CALL, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommon::Execute_Call },
	{ GE_CMD_RET, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommon::Execute_Ret },
	{ GE_CMD_END, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_End },
	{ GE_CMD_VADDR, FLAG_EXECUTE, 0, &GPUCommon::Execute_Vaddr },
	{ GE_CMD_IADDR, FLAG_EXECUTE, 0, &GPUCommon::Execute_Iaddr },
	{ GE_CMD_BJUMP, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommon::Execute_BJump },  // EXECUTE
	{ GE_CMD_BOUNDINGBOX, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_BoundingBox }, // Shouldn't need to FLUSHBEFORE.

	{ GE_CMD_PRIM, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_Prim },
	{ GE_CMD_BEZIER, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_Bezier },
	{ GE_CMD_SPLINE, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_Spline },

	// Changing the vertex type requires us to flush - except if we're soft skinning. So handled it in the execute function if needed.
	{ GE_CMD_VERTEXTYPE, FLAG_EXECUTEONCHANGE, 0, &GPUCommonHWMetal::Execute_VertexTypeSkinning },

	{ GE_CMD_LOADCLUT, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_LoadClut},

	// These two are actually processed in CMD_END.
	{ GE_CMD_SIGNAL },
	{ GE_CMD_FINISH },

	// Changes that dirty the framebuffer
	{ GE_CMD_FRAMEBUFPTR, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_FRAMEBUFWIDTH, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_VIEWPORTSCISSOR_STATE },
	{ GE_CMD_FRAMEBUFPIXFORMAT, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_BLEND_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_ZBUFPTR, FLAG_FLUSHBEFOREONCHANGE },
	{ GE_CMD_ZBUFWIDTH, FLAG_FLUSHBEFOREONCHANGE },

	{ GE_CMD_FOGCOLOR, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FOGCOLOR },
	{ GE_CMD_FOG1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FOGCOEF },
	{ GE_CMD_FOG2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FOGCOEF },

	// These affect the fragment shader so need flushing.
	{ GE_CMD_CLEARMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_RASTER_STATE | DIRTY_VIEWPORTSCISSOR_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_TEXTUREMAPENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_FOGENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_TEXMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_TEXSHADELS, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	// Raster state for Direct3D 9, uncommon.
	{ GE_CMD_SHADEMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_RASTER_STATE },
	{ GE_CMD_TEXFUNC, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAGMENTSHADER_STATE | DIRTY_TEX_ALPHA_MUL },
	{ GE_CMD_COLORTEST, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_ALPHATESTENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_COLORTESTENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_COLORTESTMASK, FLAG_FLUSHBEFOREONCHANGE, DIRTY_ALPHACOLORMASK | DIRTY_FRAGMENTSHADER_STATE },

	// These change the vertex shader so need flushing.
	{ GE_CMD_REVERSENORMAL, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_LIGHTINGENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_LIGHTENABLE0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_LIGHTENABLE1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_LIGHTENABLE2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_LIGHTENABLE3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_LIGHTTYPE0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_LIGHT0 },
	{ GE_CMD_LIGHTTYPE1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_LIGHT1 },
	{ GE_CMD_LIGHTTYPE2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_LIGHT2 },
	{ GE_CMD_LIGHTTYPE3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_LIGHT3 },
	{ GE_CMD_MATERIALUPDATE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },

	// These change all shaders so need flushing.
	{ GE_CMD_LIGHTMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },

	{ GE_CMD_TEXFILTER, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXWRAP, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS | DIRTY_FRAGMENTSHADER_STATE },

	// Uniform changes. though the fragmentshader optimizes based on these sometimes.
	{ GE_CMD_ALPHATEST, FLAG_FLUSHBEFOREONCHANGE, DIRTY_ALPHACOLORREF | DIRTY_ALPHACOLORMASK | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_COLORREF, FLAG_FLUSHBEFOREONCHANGE, DIRTY_ALPHACOLORREF | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_TEXENVCOLOR, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXENV },

	// Simple render state changes. Handled in StateMapping.cpp.
	{ GE_CMD_CULL, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_STATE },
	{ GE_CMD_CULLFACEENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_STATE },
	{ GE_CMD_DITHERENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_STATE },
	{ GE_CMD_STENCILOP, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_STENCILTEST, FLAG_FLUSHBEFOREONCHANGE, DIRTY_STENCILREPLACEVALUE | DIRTY_BLEND_STATE | DIRTY_DEPTHSTENCIL_STATE },
	{ GE_CMD_STENCILTESTENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_ALPHABLENDENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_BLENDMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_BLENDFIXEDA, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_BLENDFIXEDB, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_MASKRGB, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_COLORWRITEMASK },
	{ GE_CMD_MASKALPHA, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_DEPTHSTENCIL_STATE | DIRTY_COLORWRITEMASK },
	{ GE_CMD_ZTEST, FLAG_FLUSHBEFOREONCHANGE, DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_ZTESTENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_ZWRITEDISABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_DEPTHSTENCIL_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_LOGICOP, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_LOGICOPENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_BLEND_STATE | DIRTY_FRAGMENTSHADER_STATE },

	{ GE_CMD_TEXMAPMODE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },

	// These are read on every SubmitPrim, no need for dirtying or flushing.
	{ GE_CMD_TEXSCALEU },
	{ GE_CMD_TEXSCALEV },
	{ GE_CMD_TEXOFFSETU },
	{ GE_CMD_TEXOFFSETV },

	{ GE_CMD_TEXSIZE0, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_TexSize0 },
	{ GE_CMD_TEXSIZE1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE4, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE5, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE6, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXSIZE7, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXFORMAT, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_IMAGE },
	{ GE_CMD_TEXLEVEL, FLAG_EXECUTEONCHANGE, DIRTY_TEXTURE_PARAMS, &GPUCommonHWMetal::Execute_TexLevel },
	{ GE_CMD_TEXLODSLOPE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_IMAGE },
	{ GE_CMD_TEXADDR1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR4, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR5, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR6, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXADDR7, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_IMAGE },
	{ GE_CMD_TEXBUFWIDTH1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH4, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH5, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH6, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },
	{ GE_CMD_TEXBUFWIDTH7, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS },

	// These must flush on change, so that LoadClut doesn't have to always flush.
	{ GE_CMD_CLUTADDR, FLAG_FLUSHBEFOREONCHANGE },
	{ GE_CMD_CLUTADDRUPPER, FLAG_FLUSHBEFOREONCHANGE },
	{ GE_CMD_CLUTFORMAT, FLAG_FLUSHBEFOREONCHANGE, DIRTY_TEXTURE_PARAMS | DIRTY_DEPAL },

	// Morph weights. TODO: Remove precomputation?
	{ GE_CMD_MORPHWEIGHT0, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT1, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT2, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT3, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT4, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT5, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT6, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },
	{ GE_CMD_MORPHWEIGHT7, FLAG_FLUSHBEFOREONCHANGE | FLAG_EXECUTEONCHANGE, 0, &GPUCommon::Execute_MorphWeight },

	// Control spline/bezier patches. Don't really require flushing as such, but meh.
	{ GE_CMD_PATCHDIVISION, FLAG_FLUSHBEFOREONCHANGE },
	{ GE_CMD_PATCHPRIMITIVE, FLAG_FLUSHBEFOREONCHANGE },
	{ GE_CMD_PATCHFACING, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE },
	{ GE_CMD_PATCHCULLENABLE, FLAG_FLUSHBEFOREONCHANGE },

	// Can probably ignore this one as we don't support AA lines.
	{ GE_CMD_ANTIALIASENABLE, FLAG_FLUSHBEFOREONCHANGE },

	// Viewport.
	{ GE_CMD_OFFSETX, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_OFFSET },
	{ GE_CMD_OFFSETY, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_OFFSET },
	{ GE_CMD_VIEWPORTXSCALE, FLAG_FLUSHBEFOREONCHANGE,  DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS | DIRTY_VIEWPORTSCISSOR_STATE },  // GetFramebufferHeuristicInputs still uses viewport scale
	{ GE_CMD_VIEWPORTYSCALE, FLAG_FLUSHBEFOREONCHANGE,  DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS | DIRTY_VIEWPORTSCISSOR_STATE },
	{ GE_CMD_VIEWPORTXCENTER, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS },
	{ GE_CMD_VIEWPORTYCENTER, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS },
	{ GE_CMD_VIEWPORTZSCALE, FLAG_FLUSHBEFOREONCHANGE,  DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS },
	{ GE_CMD_VIEWPORTZCENTER, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORT_UNIFORMS },

	// Z range and clipping
	{ GE_CMD_DEPTHCLIPENABLE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_RASTER_STATE },
	{ GE_CMD_MINZ, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_STATE | DIRTY_RASTER_OFFSET | DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },
	{ GE_CMD_MAXZ, FLAG_FLUSHBEFOREONCHANGE, DIRTY_RASTER_STATE | DIRTY_RASTER_OFFSET | DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE },

	// Region
	{ GE_CMD_REGION1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_VIEWPORTSCISSOR_STATE },
	{ GE_CMD_REGION2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORTSCISSOR_STATE },

	// Scissor
	{ GE_CMD_SCISSOR1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORTSCISSOR_STATE },
	{ GE_CMD_SCISSOR2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_FRAMEBUF | DIRTY_TEXTURE_PARAMS | DIRTY_CULL_MATRIX | DIRTY_VIEWPORTSCISSOR_STATE },

	// Lighting base colors
	{ GE_CMD_AMBIENTCOLOR, FLAG_FLUSHBEFOREONCHANGE, DIRTY_AMBIENT },
	{ GE_CMD_AMBIENTALPHA, FLAG_FLUSHBEFOREONCHANGE, DIRTY_AMBIENT },
	{ GE_CMD_MATERIALDIFFUSE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATDIFFUSE },
	{ GE_CMD_MATERIALEMISSIVE, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATEMISSIVE },
	{ GE_CMD_MATERIALAMBIENT, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATAMBIENTALPHA },
	{ GE_CMD_MATERIALALPHA, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATAMBIENTALPHA },
	{ GE_CMD_MATERIALSPECULAR, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATSPECULAR },
	{ GE_CMD_MATERIALSPECULARCOEF, FLAG_FLUSHBEFOREONCHANGE, DIRTY_MATSPECULAR },

	// Light parameters
	{ GE_CMD_LX0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LY0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LZ0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LX1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LY1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LZ1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LX2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LY2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LZ2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LX3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LY3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LZ3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	{ GE_CMD_LDX0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LDY0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LDZ0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LDX1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LDY1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LDZ1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LDX2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LDY2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LDZ2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LDX3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LDY3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LDZ3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	{ GE_CMD_LKA0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LKB0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LKC0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LKA1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LKB1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LKC1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LKA2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LKB2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LKC2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LKA3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LKB3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LKC3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	{ GE_CMD_LKS0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LKS1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LKS2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LKS3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	{ GE_CMD_LKO0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LKO1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LKO2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LKO3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	{ GE_CMD_LAC0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LDC0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LSC0, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT0 },
	{ GE_CMD_LAC1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LDC1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LSC1, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT1 },
	{ GE_CMD_LAC2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LDC2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LSC2, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT2 },
	{ GE_CMD_LAC3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LDC3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },
	{ GE_CMD_LSC3, FLAG_FLUSHBEFOREONCHANGE, DIRTY_LIGHT3 },

	// Ignored commands
	{ GE_CMD_TEXFLUSH, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_TexFlush },
	{ GE_CMD_TEXSYNC, 0 },

	// These are just nop or part of other later commands.
	{ GE_CMD_NOP, 0 },
	{ GE_CMD_BASE, 0 },
	{ GE_CMD_TRANSFERSRC, 0 },
	{ GE_CMD_TRANSFERSRCW, 0 },
	{ GE_CMD_TRANSFERDST, 0 },
	{ GE_CMD_TRANSFERDSTW, 0 },
	{ GE_CMD_TRANSFERSRCPOS, 0 },
	{ GE_CMD_TRANSFERDSTPOS, 0 },
	{ GE_CMD_TRANSFERSIZE, 0 },
	{ GE_CMD_TRANSFERSTART, FLAG_EXECUTE | FLAG_READS_PC, 0, &GPUCommonHWMetal::Execute_BlockTransferStart },

	// We don't use the dither table.
	{ GE_CMD_DITH0 },
	{ GE_CMD_DITH1 },
	{ GE_CMD_DITH2 },
	{ GE_CMD_DITH3 },

	// These handle their own flushing.
	{ GE_CMD_WORLDMATRIXNUMBER, FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_WorldMtxNum },
	{ GE_CMD_WORLDMATRIXDATA,   FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_WorldMtxData },
	{ GE_CMD_VIEWMATRIXNUMBER,  FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_ViewMtxNum },
	{ GE_CMD_VIEWMATRIXDATA,    FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_ViewMtxData },
	{ GE_CMD_PROJMATRIXNUMBER,  FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_ProjMtxNum },
	{ GE_CMD_PROJMATRIXDATA,    FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_ProjMtxData },
	{ GE_CMD_TGENMATRIXNUMBER,  FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_TgenMtxNum },
	{ GE_CMD_TGENMATRIXDATA,    FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_TgenMtxData },
	{ GE_CMD_BONEMATRIXNUMBER,  FLAG_EXECUTE | FLAG_READS_PC | FLAG_WRITES_PC, 0, &GPUCommonHWMetal::Execute_BoneMtxNum },
	{ GE_CMD_BONEMATRIXDATA,    FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_BoneMtxData },

	// Vertex Screen/Texture/Color
	{ GE_CMD_VSCX },
	{ GE_CMD_VSCY },
	{ GE_CMD_VSCZ },
	{ GE_CMD_VTCS },
	{ GE_CMD_VTCT },
	{ GE_CMD_VTCQ },
	{ GE_CMD_VCV },
	{ GE_CMD_VAP, FLAG_EXECUTE, 0, &GPUCommonHWMetal::Execute_ImmVertexAlphaPrim },
	{ GE_CMD_VFC },
	{ GE_CMD_VSCV },

	// "Missing" commands (gaps in the sequence)
	{ GE_CMD_UNKNOWN_03, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_0D, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_11, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_29, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_34, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_35, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_39, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_4E, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_4F, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_52, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_59, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_5A, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_B6, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_B7, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_D1, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_ED, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_EF, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_FA, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_FB, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_FC, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_FD, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	{ GE_CMD_UNKNOWN_FE, FLAG_EXECUTE, 0, &GPUCommon::Execute_Unknown },
	// Appears to be debugging related or something?  Hit a lot in GoW.
	{ GE_CMD_NOP_FF, 0 },
};

GPUCommonHWMetal::GPUCommonHWMetal(GraphicsContext *gfxCtx, Draw::DrawContext *draw) : GPUCommon(gfxCtx, draw) {
	memset(cmdInfo_, 0, sizeof(cmdInfo_));

	// Convert the command table to a faster format, and check for dupes.
	std::set<u8> dupeCheck;
	for (size_t i = 0; i < ARRAY_SIZE(metalCommandTable); i++) {
		const u8 cmd = metalCommandTable[i].cmd;
		if (dupeCheck.find(cmd) != dupeCheck.end()) {
			ERROR_LOG(Log::G3D, "Command table Dupe: %02x (%i)", (int)cmd, (int)cmd);
		} else {
			dupeCheck.insert(cmd);
		}
		cmdInfo_[cmd].flags |= (uint64_t)metalCommandTable[i].flags | (metalCommandTable[i].dirty << 8);
		cmdInfo_[cmd].func = metalCommandTable[i].func;
		if ((cmdInfo_[cmd].flags & (FLAG_EXECUTE | FLAG_EXECUTEONCHANGE)) && !cmdInfo_[cmd].func) {
			// Can't have FLAG_EXECUTE commands without a function pointer to execute.
			Crash();
		}
	}
	// Find commands missing from the table.
	for (int i = 0; i < 0xEF; i++) {
		if (dupeCheck.find((u8)i) == dupeCheck.end()) {
			ERROR_LOG(Log::G3D, "Command missing from table: %02x (%i)", i, i);
		}
	}

	UpdateCmdInfo();
	UpdateMSAALevel(draw);
}

GPUCommonHWMetal::~GPUCommonHWMetal() {
	// Clear features so they're not visible in system info.
	gstate_c.SetUseFlags(0);

	// Delete the various common managers.
	framebufferManager_->DestroyAllFBOs();
	delete framebufferManager_;
	delete textureCache_;
	if (shaderManager_) {
		shaderManager_->ClearShaders();
		delete shaderManager_;
	}
}

// Called once per frame. Might also get called during the pause screen
// if "transparent".
void GPUCommonHWMetal::CheckConfigChanged(const DisplayLayoutConfig &config) {
	if (configChanged_) {
		ClearCacheNextFrame();
		gstate_c.SetUseFlags(CheckGPUFeatures());
		drawEngineCommon_->NotifyConfigChanged();
		textureCache_->NotifyConfigChanged();
		framebufferManager_->NotifyConfigChanged();
		BuildReportingInfo();
		configChanged_ = false;
	}
}

void GPUCommonHWMetal::CheckDisplayResized() {
	if (displayResized_) {
		framebufferManager_->NotifyDisplayResized();
		displayResized_ = false;
	}
}

void GPUCommonHWMetal::CheckRenderResized(const DisplayLayoutConfig &config) {
	if (renderResized_) {
		framebufferManager_->NotifyRenderResized(config, msaaLevel_);
		renderResized_ = false;
	}
}

// Call at the END of the GPU implementation's DeviceLost
void GPUCommonHWMetal::DeviceLost() {
	framebufferManager_->DeviceLost();
	draw_ = nullptr;
	textureCache_->Clear(false);
	textureCache_->DeviceLost();
	shaderManager_->DeviceLost();
	drawEngineCommon_->DeviceLost();
}

// Call at the start of the GPU implementation's DeviceRestore
void GPUCommonHWMetal::DeviceRestore(Draw::DrawContext *draw) {
	draw_ = draw;
	displayResized_ = true;  // re-check display bounds.
	renderResized_ = true;
	framebufferManager_->DeviceRestore(draw_);
	textureCache_->DeviceRestore(draw_);
	shaderManager_->DeviceRestore(draw_);
	drawEngineCommon_->DeviceRestore(draw_);

	PPGeSetDrawContext(draw_);

	gstate_c.SetUseFlags(CheckGPUFeatures());
	BuildReportingInfo();
	UpdateCmdInfo();
}

void GPUCommonHWMetal::UpdateCmdInfo() {
	// Reconfigure for light ubershader or not.
	for (int i = 0; i < 4; i++) {
		if (gstate_c.Use(GPU_USE_LIGHT_UBERSHADER)) {
			cmdInfo_[GE_CMD_LIGHTENABLE0 + i].RemoveDirty(DIRTY_VERTEXSHADER_STATE);
			cmdInfo_[GE_CMD_LIGHTENABLE0 + i].AddDirty(DIRTY_LIGHT_CONTROL);
			cmdInfo_[GE_CMD_LIGHTTYPE0 + i].RemoveDirty(DIRTY_VERTEXSHADER_STATE);
			cmdInfo_[GE_CMD_LIGHTTYPE0 + i].AddDirty(DIRTY_LIGHT_CONTROL);
		} else {
			cmdInfo_[GE_CMD_LIGHTENABLE0 + i].RemoveDirty(DIRTY_LIGHT_CONTROL);
			cmdInfo_[GE_CMD_LIGHTENABLE0 + i].AddDirty(DIRTY_VERTEXSHADER_STATE);
			cmdInfo_[GE_CMD_LIGHTTYPE0 + i].RemoveDirty(DIRTY_LIGHT_CONTROL);
			cmdInfo_[GE_CMD_LIGHTTYPE0 + i].AddDirty(DIRTY_VERTEXSHADER_STATE);
		}
	}

	if (gstate_c.Use(GPU_USE_LIGHT_UBERSHADER)) {
		cmdInfo_[GE_CMD_MATERIALUPDATE].RemoveDirty(DIRTY_VERTEXSHADER_STATE);
		cmdInfo_[GE_CMD_MATERIALUPDATE].AddDirty(DIRTY_LIGHT_CONTROL);
	} else {
		cmdInfo_[GE_CMD_MATERIALUPDATE].RemoveDirty(DIRTY_LIGHT_CONTROL);
		cmdInfo_[GE_CMD_MATERIALUPDATE].AddDirty(DIRTY_VERTEXSHADER_STATE);
	}
}

void GPUCommonHWMetal::SetDisplayFramebuffer(u32 framebuf, u32 stride, GEBufferFormat format) {
	framebufferManager_->SetDisplayFramebuffer(framebuf, stride, format);
	NotifyDisplay(framebuf, stride, format);
}

void GPUCommonHWMetal::CheckFlushOp(int cmd, u32 diff) {
	const u8 cmdFlags = cmdInfo_[cmd].flags;
	if (diff && (cmdFlags & FLAG_FLUSHBEFOREONCHANGE)) {
		if (dumpThisFrame_) {
			NOTICE_LOG(Log::G3D, "================ FLUSH ================");
		}
		drawEngineCommon_->Flush();
	}
}

void GPUCommonHWMetal::PreExecuteOp(u32 op, u32 diff) {
	CheckFlushOp(op >> 24, diff);
}

void GPUCommonHWMetal::PrepareCopyDisplayToOutput(const DisplayLayoutConfig &config) {
	drawEngineCommon_->FlushQueuedDepth();
	// Flush anything left over.
	drawEngineCommon_->Flush();

	// after this, render pass is active.
	framebufferManager_->PrepareCopyDisplayToOutput(config, curFramebufferDirty_);
}

void GPUCommonHWMetal::CopyDisplayToOutput(const DisplayLayoutConfig &config) {
	framebufferManager_->CopyDisplayToOutput(config);
	curFramebufferDirty_ = false;
}

bool GPUCommonHWMetal::PresentedThisFrame() const {
	return framebufferManager_->PresentedThisFrame();
}

void GPUCommonHWMetal::DoState(PointerWrap &p) {
	GPUCommon::DoState(p);

	// TODO: Some of these things may not be necessary.
	// None of these are necessary when saving.
	if (p.mode == p.MODE_READ && !PSP_CoreParameter().frozen) {
		textureCache_->Clear(true);

		gstate_c.Dirty(DIRTY_TEXTURE_IMAGE);
		framebufferManager_->DestroyAllFBOs();
	}
}

void GPUCommonHWMetal::ClearCacheNextFrame() {
	textureCache_->ClearNextFrame();
}

// Needs to be called on GPU thread, not reporting thread.
void GPUCommonHWMetal::BuildReportingInfo() {
	using namespace Draw;

	reportingPrimaryInfo_ = draw_->GetInfoString(InfoField::VENDORSTRING);
	reportingFullInfo_ = reportingPrimaryInfo_ + " - " + System_GetProperty(SYSPROP_GPUDRIVER_VERSION) + " - " + draw_->GetInfoString(InfoField::SHADELANGVERSION);
}

u32 GPUCommonHWMetal::CheckGPUFeatures() const {
	u32 features = 0;
	if (draw_->GetDeviceCaps().logicOpSupported) {
		features |= GPU_USE_LOGIC_OP;
	}
	if (draw_->GetDeviceCaps().anisoSupported && g_Config.iAnisotropyLevel > 0) {
		features |= GPU_USE_ANISOTROPY;
	}
	if (draw_->GetDeviceCaps().dualSourceBlend) {
		if (!g_Config.bVendorBugChecksEnabled || !draw_->GetBugs().Has(Draw::Bugs::DUAL_SOURCE_BLENDING_BROKEN)) {
			features |= GPU_USE_DUALSOURCE_BLEND;
		}
	}
	if (draw_->GetDeviceCaps().blendMinMaxSupported) {
		features |= GPU_USE_BLEND_MINMAX;
	}

	// The next three (clipDistance, cullDistance, depthClamp) are the critical features that let us correctly emulate PSP behavior without fallbacks.
	// This is except for the case of clipped polygons still reaching outside the guardband, where we still need to software-clip if detected.
	if (draw_->GetDeviceCaps().maxClipDistances >= 3) {
		features |= GPU_USE_CLIP_DISTANCE;
	}

	if (draw_->GetDeviceCaps().maxCullDistances >= 1) {
		features |= GPU_USE_CULL_DISTANCE;
	}

	if (draw_->GetDeviceCaps().fragmentShaderFullPrecisionFloat) {
		features |= GPU_USE_FULL_PRECISION_IN_FRAGMENT;
	}

	if (draw_->GetDeviceCaps().depthClampSupported) {
		features |= GPU_USE_DEPTH_CLAMP;
	}

	if (draw_->GetDeviceCaps().textureDepthSupported) {
		features |= GPU_USE_DEPTH_TEXTURE;
	}

	if (draw_->GetDeviceCaps().samplerLodControl) {
		features |= GPU_USE_SAMPLER_LOD_CONTROL;
		if (!draw_->GetDeviceCaps().samplerLodBias) {
			features |= GPU_USE_SHADER_LOD_BIAS;
		}
	}

	if (draw_->GetDeviceCaps().framebufferFetchSupported) {
		features |= GPU_USE_FRAMEBUFFER_FETCH;
		features |= GPU_USE_SHADER_BLENDING;   // doesn't matter if we are buffered or not here.
	} else {
		if (!g_Config.bSkipBufferEffects) {
			features |= GPU_USE_SHADER_BLENDING;
		}
	}

	if (draw_->GetShaderLanguageDesc().bitwiseOps && g_Config.bUberShaderVertex) {
		features |= GPU_USE_LIGHT_UBERSHADER;
	}

	if (PSP_CoreParameter().compat.flags().ClearToRAM) {
		features |= GPU_USE_CLEAR_RAM_HACK;
	}

	if (!PSP_CoreParameter().compat.flags().DisableRangeCulling && !draw_->GetBugs().Has(Draw::Bugs::BROKEN_NAN_IN_CONDITIONAL)) {
		features |= GPU_USE_VS_RANGE_CULLING;
	}

	// Set this to true to easily test our fallback shaders for depth clamp and min/max clipping.
	// DO NOT check that in! This should be false normally.
	constexpr bool simulateOldMali = false;
	if (simulateOldMali) {
		features &= ~(GPU_USE_CLIP_DISTANCE | GPU_USE_CULL_DISTANCE | GPU_USE_DEPTH_CLAMP);
	}

	return features;
}

u32 GPUCommonHWMetal::CheckGPUFeaturesLate(u32 features) const {
	// If we already have a 16-bit depth buffer, we don't need to round.
	bool prefer16 = draw_->GetDeviceCaps().preferredDepthBufferFormat == Draw::DataFormat::D16;
	if (!prefer16) {
		if (PSP_CoreParameter().compat.flags().PixelDepthRounding) {
			// Use fragment rounding on where available otherwise.
			features |= GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT;
		} else if (PSP_CoreParameter().compat.flags().VertexDepthRounding) {
			features |= GPU_ROUND_DEPTH_TO_16BIT;
		}
	}
	return features;
}

void GPUCommonHWMetal::UpdateMSAALevel(Draw::DrawContext *draw) {
	int level = g_Config.iMultiSampleLevel;
	if (draw && draw->GetDeviceCaps().multiSampleLevelsMask & (1 << level)) {
		msaaLevel_ = level;
	} else {
		// Didn't support the configured level, so revert to 0.
		msaaLevel_ = 0;
	}
}

std::vector<std::string> GPUCommonHWMetal::DebugGetShaderIDs(DebugShaderType type) {
	switch (type) {
	case SHADER_TYPE_VERTEXLOADER:
		return drawEngineCommon_->DebugGetVertexLoaderIDs();
	case SHADER_TYPE_TEXTURE:
		return textureCache_->GetTextureShaderCache().DebugGetShaderIDs(type);
	default:
		return shaderManager_->DebugGetShaderIDs(type);
	}
}

std::string GPUCommonHWMetal::DebugGetShaderString(std::string id, DebugShaderType type, DebugShaderStringType stringType) {
	switch (type) {
	case SHADER_TYPE_VERTEXLOADER:
		return drawEngineCommon_->DebugGetVertexLoaderString(id, stringType);
	case SHADER_TYPE_TEXTURE:
		return textureCache_->GetTextureShaderCache().DebugGetShaderString(id, type, stringType);
	default:
		return shaderManager_->DebugGetShaderString(id, type, stringType);
	}
}

bool GPUCommonHWMetal::GetCurrentFramebuffer(GPUDebugBuffer &buffer, GPUDebugFramebufferType type, int maxRes) {
	if (type == GPU_DBG_FRAMEBUF_DISPLAY) {
		return framebufferManager_->GetDisplayFramebuffer(buffer, maxRes);
	}
	u32 fb_address = type == GPU_DBG_FRAMEBUF_RENDER ? (gstate.getFrameBufRawAddress() | 0x04000000) : framebufferManager_->DisplayFramebufAddr();
	int fb_stride = type == GPU_DBG_FRAMEBUF_RENDER ? gstate.FrameBufStride() : framebufferManager_->DisplayFramebufStride();
	GEBufferFormat format = type == GPU_DBG_FRAMEBUF_RENDER ? gstate_c.framebufFormat : framebufferManager_->DisplayFramebufFormat();
	return framebufferManager_->GetFramebuffer(fb_address, fb_stride, format, buffer, maxRes);
}

bool GPUCommonHWMetal::GetCurrentDepthbuffer(GPUDebugBuffer &buffer) {
	u32 fb_address = gstate.getFrameBufRawAddress() | 0x04000000;
	int fb_stride = gstate.FrameBufStride();

	u32 z_address = gstate.getDepthBufRawAddress() | 0x04000000;
	int z_stride = gstate.DepthBufStride();

	return framebufferManager_->GetDepthbuffer(fb_address, fb_stride, z_address, z_stride, buffer);
}

bool GPUCommonHWMetal::GetCurrentStencilbuffer(GPUDebugBuffer &buffer) {
	u32 fb_address = gstate.getFrameBufRawAddress() | 0x04000000;
	int fb_stride = gstate.FrameBufStride();

	return framebufferManager_->GetStencilbuffer(fb_address, fb_stride, buffer);
}

bool GPUCommonHWMetal::GetOutputFramebuffer(GPUDebugBuffer &buffer) {
	// framebufferManager_ can be null here when taking screens in software rendering mode.
	// TODO: Actually grab the framebuffer anyway.
	return framebufferManager_ ? framebufferManager_->GetOutputFramebuffer(buffer) : false;
}

std::vector<const VirtualFramebuffer *> GPUCommonHWMetal::GetFramebufferList() const {
	return framebufferManager_->GetFramebufferList();
}

bool GPUCommonHWMetal::GetCurrentClut(GPUDebugBuffer &buffer) {
	return textureCache_->GetCurrentClutBuffer(buffer);
}

bool GPUCommonHWMetal::GetCurrentTexture(GPUDebugBuffer &buffer, int level, bool *isFramebuffer) {
	if (!gstate.isTextureMapEnabled()) {
		return false;
	}
	return textureCache_->GetCurrentTextureDebug(buffer, level, isFramebuffer);
}

void GPUCommonHWMetal::CheckDepthUsage(VirtualFramebuffer *vfb) {
	if (!gstate_c.usingDepth) {
		bool isReadingDepth = false;
		bool isClearingDepth = false;
		bool isWritingDepth = false;
		if (gstate.isModeClear()) {
			isClearingDepth = gstate.isClearModeDepthMask();
			isWritingDepth = isClearingDepth;
		} else if (gstate.isDepthTestEnabled()) {
			isWritingDepth = gstate.isDepthWriteEnabled();
			isReadingDepth = gstate.getDepthTestFunction() > GE_COMP_ALWAYS;
		}

		if (isWritingDepth || isReadingDepth) {
			gstate_c.usingDepth = true;
			gstate_c.clearingDepth = isClearingDepth;
			auto markDepthUsed = [&]() {
				vfb->last_frame_depth_render = gpuStats.totals.numFlips;
				if (isWritingDepth) {
					vfb->last_frame_depth_updated = gpuStats.totals.numFlips;
				}
			};
			// SetDepthFrameBuffer copies in depth from a framebuffer sharing it, if that's newer than this one's.
			// A pass that tests against the depth it finds gets that checked before this buffer's depth counts as
			// used this frame: Brooktown High clears depth through another framebuffer as the last thing in a frame,
			// and the next frame tests against it (#7223). A pass that only writes depth doesn't need the copy.
			if (!isReadingDepth) {
				markDepthUsed();
			}
			framebufferManager_->SetDepthFrameBuffer(isClearingDepth);
			markDepthUsed();
		}
	}
}

void GPUCommonHWMetal::InvalidateCache(u32 addr, int size, GPUInvalidationType type) {
	if (size > 0)
		textureCache_->Invalidate(addr, size, type);
	else
		textureCache_->InvalidateAll(type);

	if (type != GPU_INVALIDATE_ALL && framebufferManager_->MayIntersectFramebufferColor(addr)) {
		// Vempire invalidates (with writeback) after drawing, but before blitting.
		// TODO: Investigate whether we can get this to work some other way.
		if (type == GPU_INVALIDATE_SAFE) {
			framebufferManager_->UpdateFromMemory(addr, size);
		}
	}
}

bool GPUCommonHWMetal::FramebufferDirty() {
	if (!framebufferManager_)
		return true;
	VirtualFramebuffer *vfb = framebufferManager_->GetDisplayVFB();
	if (vfb) {
		bool dirty = vfb->dirtyAfterDisplay;
		vfb->dirtyAfterDisplay = false;
		return dirty;
	}
	return true;
}

bool GPUCommonHWMetal::FramebufferReallyDirty() {
	if (!framebufferManager_)
		return true;
	VirtualFramebuffer *vfb = framebufferManager_->GetDisplayVFB();
	if (vfb) {
		bool dirty = vfb->reallyDirtyAfterDisplay;
		vfb->reallyDirtyAfterDisplay = false;
		return dirty;
	}
	return true;
}

void GPUCommonHWMetal::ExecuteOp(u32 op, u32 diff) {
	const u8 cmd = op >> 24;
	const MetalCommandInfo info = cmdInfo_[cmd];
	const u8 cmdFlags = info.flags;
	if ((cmdFlags & FLAG_EXECUTE) || (diff && (cmdFlags & FLAG_EXECUTEONCHANGE))) {
		(this->*info.func)(op, diff);
	} else if (diff) {
		uint64_t dirty = info.flags >> 8;
		if (dirty)
			gstate_c.Dirty(dirty);
	}
}

void GPUCommonHWMetal::FastRunLoop(DisplayList &list) {
	PROFILE_THIS_SCOPE("gpuloop");

	if (!Memory::IsValidAddress(list.pc)) {
		// We're having some serious problems here, just bail and try to limp along and not crash the app.
		downcount = 0;
		return;
	}

	const MetalCommandInfo *cmdInfo = cmdInfo_;
	int dc = downcount;
	for (; dc > 0; --dc) {
		// We know that display list PCs have the upper nibble == 0 - no need to mask the pointer
		const u32 op = *(const u32_le *)(Memory::base + list.pc);
		const u32 cmd = op >> 24;
		const MetalCommandInfo &info = cmdInfo[cmd];
		const u32 diff = op ^ gstate.cmdmem[cmd];
		if (diff == 0) {
			if (info.flags & FLAG_EXECUTE) {
				downcount = dc;
				(this->*info.func)(op, diff);
				dc = downcount;
			}
		} else {
			uint64_t flags = info.flags;
			if (flags & FLAG_FLUSHBEFOREONCHANGE) {
				drawEngineCommon_->Flush();
			}
			gstate.cmdmem[cmd] = op;
			if (flags & (FLAG_EXECUTE | FLAG_EXECUTEONCHANGE)) {
				downcount = dc;
				(this->*info.func)(op, diff);
				dc = downcount;
			} else {
				uint64_t dirty = flags >> 8;
				if (dirty)
					gstate_c.Dirty(dirty);
			}
		}
		list.pc += 4;
	}
	downcount = 0;
}

void GPUCommonHWMetal::Execute_VertexTypeSkinning(u32 op, u32 diff) {
	// Don't flush when weight count changes.
	if (diff & ~GE_VTYPE_WEIGHTCOUNT_MASK) {
		// Restore and flush
		gstate.vertType ^= diff;
		Flush();
		gstate.vertType ^= diff;
		gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE);
	}

	if (diff & GE_VTYPE_THROUGH_MASK)
		gstate_c.Dirty(DIRTY_RASTER_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_VIEWPORTSCISSOR_STATE | DIRTY_FRAGMENTSHADER_STATE);
}

void GPUCommonHWMetal::Execute_Prim(u32 op, u32 diff) {
	// This drives all drawing. All other state we just buffer up, then we apply it only
	// when it's time to draw. As most PSP games set state redundantly ALL THE TIME, this is a huge optimization.

	PROFILE_THIS_SCOPE("execprim");

	FlushImm();
	UpdateMatrixProducts();

	// Upper bits are ignored.
	const GEPrimitiveType prim = static_cast<GEPrimitiveType>((op >> 16) & 7);
	SetDrawType(DRAW_PRIM, prim);

	// Discard AA lines as we can't do anything that makes sense with these anyway. The SW plugin might, though.
	if (gstate.isAntiAliasEnabled()) {
		// Heuristic derived from discussions in #6483 and #12588.
		// Discard AA lines in Persona 3 Portable, DOA Paradise and Summon Night 5, while still keeping AA lines in Echochrome.
		if ((prim == GE_PRIM_LINE_STRIP || prim == GE_PRIM_LINES) && gstate.getTextureFunction() == GE_TEXFUNC_REPLACE)
			return;
	}

	// Update cached framebuffer format.
	// We store it in the cache so it can be modified for blue-to-alpha, next.
	gstate_c.framebufFormat = gstate.FrameBufFormat();

	if (!Memory::IsValidAddress(gstate_c.vertexAddr)) {
		ERROR_LOG(Log::G3D, "Bad vertex address %08x!", gstate_c.vertexAddr);
		return;
	}

	// See the documentation for gstate_c.blueToAlpha.
	bool blueToAlpha = false;
	if (PSP_CoreParameter().compat.flags().BlueToAlpha) {
		if (gstate_c.framebufFormat == GEBufferFormat::GE_FORMAT_565 && gstate.getColorMask() == 0x0FFFFF && !gstate.isLogicOpEnabled()) {
			blueToAlpha = true;
			gstate_c.framebufFormat = GEBufferFormat::GE_FORMAT_4444;
		}
		if (blueToAlpha != gstate_c.blueToAlpha) {
			gstate_c.blueToAlpha = blueToAlpha;
			gstate_c.Dirty(DIRTY_FRAMEBUF | DIRTY_FRAGMENTSHADER_STATE | DIRTY_BLEND_STATE);
		}
	}

	if (PSP_CoreParameter().compat.flags().SplitFramebufferMargin) {
		switch (gstate.vertType & 0xFFFFFF) {
		case 0x00800102:  // through, u16 uv, u16 pos (used for the framebuffer effect in-game)
		case 0x0080011c:  // through, 8888 color, s16 pos (used for clearing in the margin of the title screen)
		case 0x00000183:  // float uv, float pos (used for drawing in the margin of the title screen)
			// Need to re-check the framebuffer every one of these draws, to update the split if needed.
			gstate_c.Dirty(DIRTY_FRAMEBUF);
		}
	}

	// This also makes skipping drawing very effective.
	bool changed;
	VirtualFramebuffer *const vfb = framebufferManager_->SetRenderFrameBuffer(gstate_c.IsDirty(DIRTY_FRAMEBUF), gstate_c.skipDrawReason, &changed);
	if (blueToAlpha) {
		vfb->usageFlags |= FB_USAGE_BLUE_TO_ALPHA;
	}

	if (changed) {
		drawEngineCommon_->FlushQueuedDepth();
	}

	if (gstate_c.dirty & DIRTY_VERTEXSHADER_STATE) {
		vertexCost_ = EstimatePerVertexCost();
	}

	const u32 count = op & 0xFFFF;
	// Must check this after SetRenderFrameBuffer so we know SKIPDRAW_NON_DISPLAYED_FB.
	if (gstate_c.skipDrawReason & (SKIPDRAW_SKIPFRAME | SKIPDRAW_NON_DISPLAYED_FB)) {
		// Rough estimate, not sure what's correct.
		cyclesExecuted += vertexCost_ * count;
		if (gstate.isModeClear()) {
			gpuStats.perFrame.numClears++;
		}
		return;
	}

	CheckDepthUsage(vfb);

	const void *verts = Memory::GetPointerUnchecked(gstate_c.vertexAddr);
	const void *inds = nullptr;

	const bool isTriangle = IsTrianglePrim(prim);

	bool canExtend = isTriangle;
	GEVertexType vertexType = (GEVertexType)gstate.vertType;
	if ((vertexType & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
		u32 indexAddr = gstate_c.indexAddr;
		const int indexShift = ((vertexType & GE_VTYPE_IDX_MASK) >> GE_VTYPE_IDX_SHIFT) - 1;
		if (!Memory::IsValidRange(indexAddr, count << indexShift)) {
			ERROR_LOG(Log::G3D, "Bad index address %08x (%d)!", indexAddr, count);
			return;
		}
		inds = Memory::GetPointerUnchecked(indexAddr);
		canExtend = false;
	}

	// cull mode
	int cullMode = gstate.getCullMode();

	uint32_t vertTypeID = GetVertTypeID(vertexType, gstate.getUVGenMode());
	VertexDecoder *decoder = drawEngineCommon_->GetVertexDecoder(vertTypeID);

	// This is the check from #21678 for Evangelion JO.
	// TODO: Make this less specific.
	if (gstate.isModeThrough() && gstate.isTextureMapEnabled() && gstate.getColorMask() != 0xFFFFFFFF &&
		gstate.getScissorX1() == 0 && gstate.getScissorY1() == 0 && Memory::IsVRAMAddress(gstate.getFrameBufAddress())) {
		int safeWidth;
		int safeHeight;
		// First-frame readback can happen before queued draws reach backend transform.
		if (drawEngineCommon_->EstimateThroughPrimSafeSize(verts, inds, prim, count, decoder, vertexType, &safeWidth, &safeHeight)) {
			framebufferManager_->SetSafeSize(safeWidth, safeHeight);
		}
	}

	// Through mode early-out. Very beneficial for Fate Extra CCC (very beneficial there due to avoiding texture loads)
	// Also we take the opportunity to check for flat draws, where we can detect sprites (to fix filter artifacts).
	ClipInfoFlags flags{};
	if (gstate.isModeThrough()) {
		int bytesRead = 0;
		if (!drawEngineCommon_->TestBoundingBoxThrough(prim, verts, inds, count, decoder, vertexType, &bytesRead, &flags)) {
			gpuStats.perFrame.numCulledDraws++;
			int cycles = vertexCost_ * count;
			gpuStats.perFrame.vertexGPUCycles += cycles;
			cyclesExecuted += cycles;
			// We still have to advance vertex/index pointers!
			gstate_c.AdvanceVerts(vertexType, count, bytesRead);
			return;
		}
		cyclesExecuted += EstimateFillCycles(prim, verts, inds, count, decoder, vertexType);
	}

#define MAX_CULL_CHECK_COUNT 2500

	// For now, turn off culling on platforms where we don't have SIMD bounding box tests, like RISC-V.
#if PPSSPP_ARCH(ARM_NEON) || PPSSPP_ARCH(SSE2)

#define PASSES_CULLING ((vertexType & (GE_VTYPE_THROUGH_MASK | GE_VTYPE_MORPHCOUNT_MASK | GE_VTYPE_WEIGHT_MASK)) || count > MAX_CULL_CHECK_COUNT)

#else

#define PASSES_CULLING true

#endif

	// If certain conditions are true, do frustum culling.
	bool passCulling = PASSES_CULLING;
	if (!passCulling) {
		// Do software culling.
		if (drawEngineCommon_->TestBoundingBoxFast(gstate_c.cullMatrix, verts, inds, count, decoder, vertexType, &flags)) {
			passCulling = true;
		} else {
			gpuStats.perFrame.numCulledDraws++;
		}
	}

	int bytesRead = 0;

	// If the first one in a batch passes, let's assume the whole batch passes.
	// Cuts down on checking, while not losing that much efficiency.
	bool onePassed = false;
	if (passCulling) {
		if (!drawEngineCommon_->SubmitPrim(verts, inds, prim, count, decoder, vertTypeID, true, &bytesRead, flags)) {
			canExtend = false;
		}
		// onePassed = true;
	} else {
		// Still need to advance bytesRead.
		drawEngineCommon_->SkipPrim(prim, count, decoder, &bytesRead);
		canExtend = false;
	}

	// After drawing, we advance the vertexAddr (when non indexed) or indexAddr (when indexed).
	// Some games rely on this, they don't bother reloading VADDR and IADDR.
	// The VADDR/IADDR registers are NOT updated.
	gstate_c.AdvanceVerts(vertexType, count, bytesRead);

	int totalVertCount = count;

	// PRIMs are often followed by more PRIMs. Save some work and submit them immediately.
	const u32_le *start = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	const u32_le *src = start;
	const u32_le *stall = currentList->stall ? (const u32_le *)Memory::GetPointerUnchecked(currentList->stall) : 0;

	// Optimized submission of sequences of PRIM. Allows us to avoid going through all the mess
	// above for each one. This can be expanded to support additional games that intersperse
	// PRIM commands with other commands. A special case is Earth Defence Force 2 that changes culling mode
	// between each prim, we just change the triangle winding right here to still be able to join draw calls.
	constexpr uint32_t vtypeCheckMask = ~GE_VTYPE_WEIGHTCOUNT_MASK;

	if (!useFastRunLoop_)
		goto bail;  // we're either recording or stepping.

	while (src != stall) {
		uint32_t data = *src;
		switch (data >> 24) {
		case GE_CMD_PRIM:
		{
			GEPrimitiveType newPrim = static_cast<GEPrimitiveType>((data >> 16) & 7);
			if (IsTrianglePrim(newPrim) != isTriangle)
				goto bail;  // Can't join over this boundary. Might as well exit and get this on the next time around.
			// TODO: more efficient updating of verts/inds

			u32 count = data & 0xFFFF;
			bool clockwise = !gstate.isCullEnabled() || gstate.getCullMode() == cullMode;
			if (canExtend) {
				// Non-indexed draws can be cheaply merged if vertexAddr hasn't changed, that means the vertices
				// are consecutive in memory. We also ignore culling here which is not ideal.
				_dbg_assert_((vertexType & GE_VTYPE_IDX_MASK) == GE_VTYPE_IDX_NONE);
				int commandsExecuted = drawEngineCommon_->ExtendNonIndexedPrim(src, stall, decoder, vertTypeID, clockwise, &bytesRead, isTriangle, flags);
				if (!commandsExecuted) {
					goto bail;
				}
				src += commandsExecuted - 1;
				gstate_c.vertexAddr += bytesRead;
				totalVertCount += count;
				break;
			}

			verts = Memory::GetPointerUnchecked(gstate_c.vertexAddr);
			inds = nullptr;
			if ((vertexType & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
				const u32 indexAddr = gstate_c.indexAddr;
				const int indexShift = ((vertexType & GE_VTYPE_IDX_MASK) >> GE_VTYPE_IDX_SHIFT) - 1;
				if (!Memory::IsValidRange(gstate_c.indexAddr, count << indexShift)) {
					// Bad index range. Let's give up the fast loop.
					goto bail;
				}
				inds = Memory::GetPointerUnchecked(indexAddr);
			} else {
				// We can extend again after submitting a normal draw.
				canExtend = isTriangle;
			}

			bool passCulling = onePassed || PASSES_CULLING;
			ClipInfoFlags flags{};
			if (!passCulling) {
				// Do software culling.
				if (drawEngineCommon_->TestBoundingBoxFast(gstate_c.cullMatrix, verts, inds, count, decoder, vertexType, &flags)) {
					passCulling = true;
				} else {
					gpuStats.perFrame.numCulledDraws++;
				}
			}
			if (passCulling) {
				if (vertexType & GE_VTYPE_THROUGH_MASK) {
					cyclesExecuted += EstimateFillCycles(newPrim, verts, inds, count, decoder, vertexType);
				}
				if (!drawEngineCommon_->SubmitPrim(verts, inds, newPrim, count, decoder, vertTypeID, clockwise, &bytesRead, flags)) {
					canExtend = false;
				}
				// As soon as one passes, assume we don't need to check the rest of this batch.
				// onePassed = true;
			} else {
				// Still need to advance bytesRead.
				drawEngineCommon_->SkipPrim(newPrim, count, decoder, &bytesRead);
				canExtend = false;
			}
			gstate_c.AdvanceVerts(vertexType, count, bytesRead);
			totalVertCount += count;
			break;
		}
		case GE_CMD_VERTEXTYPE:
		{
			uint32_t diff = data ^ vertexType;
			// don't mask upper bits, vertexType is unmasked
			if (diff) {
				if (diff & vtypeCheckMask)
					goto bail;
				drawEngineCommon_->FlushSkin();
				canExtend = false;  // TODO: Might support extending between some vertex types in the future.
				vertexType = (GEVertexType)(data);
				vertTypeID = GetVertTypeID(vertexType, gstate.getUVGenMode());
				decoder = drawEngineCommon_->GetVertexDecoder(vertTypeID);
			}
			break;
		}
		case GE_CMD_VADDR:
		{
			gstate.cmdmem[GE_CMD_VADDR] = data;
			uint32_t newAddr = gstate_c.getRelativeAddress(data & 0x00FFFFFF);
			if (gstate_c.vertexAddr != newAddr) {
				canExtend = false;
				gstate_c.vertexAddr = newAddr;
			}
			break;
		}
		case GE_CMD_IADDR:
			gstate.cmdmem[GE_CMD_IADDR] = data;
			gstate_c.indexAddr = gstate_c.getRelativeAddress(data & 0x00FFFFFF);
			break;
		case GE_CMD_OFFSETADDR:
			gstate.cmdmem[GE_CMD_OFFSETADDR] = data;
			gstate_c.offsetAddr = data << 8;
			break;
		case GE_CMD_BASE:
			gstate.cmdmem[GE_CMD_BASE] = data;
			break;
		case GE_CMD_CULLFACEENABLE:
			// Earth Defence Force 2
			if (gstate.cmdmem[GE_CMD_CULLFACEENABLE] != data) {
				goto bail;
			}
			break;
		case GE_CMD_CULL:
			// flip face by indices for triangles
			cullMode = data & 1;
			break;
		case GE_CMD_TEXFLUSH:
		case GE_CMD_NOP:
		case GE_CMD_NOP_FF:
			gstate.cmdmem[data >> 24] = data;
			break;
		case GE_CMD_BONEMATRIXNUMBER:
			gstate.cmdmem[GE_CMD_BONEMATRIXNUMBER] = data;
			break;
		case GE_CMD_TEXSCALEU:
			// We don't "dirty-check" - we could avoid getFloat24 and setting canExtend=false, but usually
			// when texscale commands are in line with the prims like this, they actually have an effect
			// and requires us to stop extending strips anyway.
			gstate.cmdmem[GE_CMD_TEXSCALEU] = data;
			canExtend = false;
			break;
		case GE_CMD_TEXSCALEV:
			gstate.cmdmem[GE_CMD_TEXSCALEV] = data;
			canExtend = false;
			break;
		case GE_CMD_TEXOFFSETU:
			gstate.cmdmem[GE_CMD_TEXOFFSETU] = data;
			canExtend = false;
			break;
		case GE_CMD_TEXOFFSETV:
			gstate.cmdmem[GE_CMD_TEXOFFSETV] = data;
			canExtend = false;
			break;
		case GE_CMD_TEXLEVEL:
			// Same Gran Turismo hack from Execute_TexLevel
			if ((data & 3) != GE_TEXLEVEL_MODE_AUTO && (0x00FF0000 & data) != 0) {
				goto bail;
			}
			if ((data ^ gstate.cmdmem[GE_CMD_TEXLEVEL]) & 0x00FF0000) {
				// Like Execute_TexLevel. The bias is a uniform with GPU_USE_SHADER_LOD_BIAS.
				gstate_c.Dirty(DIRTY_MIPBIAS);
			}
			gstate.cmdmem[GE_CMD_TEXLEVEL] = data;
			break;
		case GE_CMD_CALL:
		{
			// A bone matrix probably. If not we bail.
			const u32 target = gstate_c.getRelativeAddress(data & 0x00FFFFFC);
			if ((Memory::ReadUnchecked_U32(target) >> 24) == GE_CMD_BONEMATRIXDATA &&
				(Memory::ReadUnchecked_U32(target + 11 * 4) >> 24) == GE_CMD_BONEMATRIXDATA &&
				(Memory::ReadUnchecked_U32(target + 12 * 4) >> 24) == GE_CMD_RET &&
				(target > currentList->stall || target + 12 * 4 < currentList->stall) &&
				(gstate.boneMatrixNumber & 0x00FFFFFF) <= 96 - 12) {
				drawEngineCommon_->FlushSkin();
				canExtend = false;
				FastLoadBoneMatrix(target);
			} else {
				goto bail;
			}
			break;
		}

		// Keep going if these commands don't change state.
		case GE_CMD_TEXBUFWIDTH0:
		case GE_CMD_TEXADDR0:
			if (data != gstate.cmdmem[data >> 24])
				goto bail;
			break;

		default:
			// All other commands might need a flush or something, stop this inner loop.
			goto bail;
		}
		src++;
	}

bail:
	drawEngineCommon_->FlushSkin();
	gstate.cmdmem[GE_CMD_VERTEXTYPE] = vertexType;
	const int cmdCount = src - start;
	// Skip over the commands we just read out manually.
	if (cmdCount > 0) {
		currentList->pc += cmdCount * 4;
		UpdatePC(currentList->pc, currentList->pc);
		// flush back cull mode
		if (cullMode != gstate.getCullMode()) {
			// We rewrote everything to the old cull mode, so flush first.
			drawEngineCommon_->Flush();

			// Now update things for next time.
			gstate.cmdmem[GE_CMD_CULL] ^= 1;
			gstate_c.Dirty(DIRTY_RASTER_STATE);
		}
	}

	int cycles = vertexCost_ * totalVertCount;
	gpuStats.perFrame.vertexGPUCycles += cycles;

	if (!PSP_CoreParameter().compat.flags().FastEmulatedGPU) {
		cyclesExecuted += cycles;
	}
}

void GPUCommonHWMetal::Execute_Bezier(u32 op, u32 diff) {
	UpdateMatrixProducts();

	// We don't dirty on normal changes anymore as we prescale, but it's needed for splines/bezier.
	gstate_c.framebufFormat = gstate.FrameBufFormat();

	// This also make skipping drawing very effective.
	bool changed;
	VirtualFramebuffer *vfb = framebufferManager_->SetRenderFrameBuffer(gstate_c.IsDirty(DIRTY_FRAMEBUF), gstate_c.skipDrawReason, &changed);
	if (changed) {
		drawEngineCommon_->FlushQueuedDepth();
	}
	if (gstate_c.skipDrawReason & (SKIPDRAW_SKIPFRAME | SKIPDRAW_NON_DISPLAYED_FB)) {
		// TODO: Should this eat some cycles?  Probably yes.  Not sure if important.
		return;
	}

	CheckDepthUsage(vfb);

	if (!Memory::IsValidAddress(gstate_c.vertexAddr)) {
		ERROR_LOG(Log::G3D, "Bad vertex address %08x!", gstate_c.vertexAddr);
		return;
	}

	const void *control_points = Memory::GetPointerUnchecked(gstate_c.vertexAddr);
	const void *indices = NULL;
	const u32 vertType = gstate.vertType;
	if ((vertType & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
		if (!Memory::IsValidAddress(gstate_c.indexAddr)) {
			ERROR_LOG(Log::G3D, "Bad index address %08x!", gstate_c.indexAddr);
			return;
		}
		indices = Memory::GetPointerUnchecked(gstate_c.indexAddr);
	}

	if (vertTypeIsSkinningEnabled(vertType)) {
		DEBUG_LOG_REPORT(Log::G3D, "Unusual bezier/spline vtype: %08x, morph: %d, bones: %d", vertType, (vertType & GE_VTYPE_MORPHCOUNT_MASK) >> GE_VTYPE_MORPHCOUNT_SHIFT, vertTypeGetNumBoneWeights(vertType));
	}

	// Can't flush after setting gstate_c.submitType below since it'll be a mess - it must be done already.
	if (flushOnParams_)
		drawEngineCommon_->Flush();

	Spline::BezierSurface surface;
	surface.tess_u = gstate.getPatchDivisionU();
	surface.tess_v = gstate.getPatchDivisionV();
	surface.num_points_u = op & 0xFF;
	surface.num_points_v = (op >> 8) & 0xFF;
	surface.num_patches_u = (surface.num_points_u - 1) / 3;
	surface.num_patches_v = (surface.num_points_v - 1) / 3;
	surface.primType = gstate.getPatchPrimitiveType();
	surface.patchFacing = gstate.patchfacing & 1;
	surface.geExact = PSP_CoreParameter().compat.flags().AccurateSplines;

	SetDrawType(DRAW_BEZIER, PatchPrimToPrim(surface.primType));

	// We need to dirty UVSCALEOFFSET here because we look at the submit type when setting that uniform.
	gstate_c.Dirty(DIRTY_RASTER_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_UVSCALEOFFSET);
	gstate_c.submitType = SubmitType::BEZIER;

	int bytesRead = 0;
	drawEngineCommon_->SubmitCurve(control_points, indices, surface, gstate.vertType, &bytesRead, "bezier");

	gstate_c.Dirty(DIRTY_RASTER_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_UVSCALEOFFSET);
	gstate_c.submitType = SubmitType::DRAW;

	// After drawing, we advance pointers - see SubmitPrim which does the same.
	const int count = surface.num_points_u * surface.num_points_v;
	gstate_c.AdvanceVerts(vertType, count, bytesRead);
}

void GPUCommonHWMetal::Execute_Spline(u32 op, u32 diff) {
	UpdateMatrixProducts();

	// We don't dirty on normal changes anymore as we prescale, but it's needed for splines/bezier.
	gstate_c.framebufFormat = gstate.FrameBufFormat();

	// This also make skipping drawing very effective.
	bool changed;
	VirtualFramebuffer *vfb = framebufferManager_->SetRenderFrameBuffer(gstate_c.IsDirty(DIRTY_FRAMEBUF), gstate_c.skipDrawReason, &changed);
	if (changed) {
		drawEngineCommon_->FlushQueuedDepth();
	}
	if (gstate_c.skipDrawReason & (SKIPDRAW_SKIPFRAME | SKIPDRAW_NON_DISPLAYED_FB)) {
		// TODO: Should this eat some cycles?  Probably yes.  Not sure if important.
		return;
	}

	CheckDepthUsage(vfb);

	if (!Memory::IsValidAddress(gstate_c.vertexAddr)) {
		ERROR_LOG(Log::G3D, "Bad vertex address %08x!", gstate_c.vertexAddr);
		return;
	}

	const void *control_points = Memory::GetPointerUnchecked(gstate_c.vertexAddr);
	const void *indices = NULL;
	const u32 vertType = gstate.vertType;
	if ((vertType & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
		if (!Memory::IsValidAddress(gstate_c.indexAddr)) {
			ERROR_LOG(Log::G3D, "Bad index address %08x!", gstate_c.indexAddr);
			return;
		}
		indices = Memory::GetPointerUnchecked(gstate_c.indexAddr);
	}

	if (vertTypeIsSkinningEnabled(vertType)) {
		WARN_LOG_ONCE(unusualcurve, Log::G3D, "Unusual bezier/spline vtype: %08x, morph: %d, bones: %d", vertType, (vertType & GE_VTYPE_MORPHCOUNT_MASK) >> GE_VTYPE_MORPHCOUNT_SHIFT, vertTypeGetNumBoneWeights(vertType));
	}

	// Can't flush after setting gstate_c.submitType below since it'll be a mess - it must be done already.
	if (flushOnParams_)
		drawEngineCommon_->Flush();

	Spline::SplineSurface surface;
	surface.tess_u = gstate.getPatchDivisionU();
	surface.tess_v = gstate.getPatchDivisionV();
	surface.type_u = (op >> 16) & 0x3;
	surface.type_v = (op >> 18) & 0x3;
	surface.num_points_u = op & 0xFF;
	surface.num_points_v = (op >> 8) & 0xFF;
	surface.num_patches_u = surface.num_points_u - 3;
	surface.num_patches_v = surface.num_points_v - 3;
	surface.primType = gstate.getPatchPrimitiveType();
	surface.patchFacing = gstate.patchfacing & 1;
	surface.geExact = PSP_CoreParameter().compat.flags().AccurateSplines;

	SetDrawType(DRAW_SPLINE, PatchPrimToPrim(surface.primType));

	// We need to dirty UVSCALEOFFSET here because we look at the submit type when setting that uniform.
	gstate_c.Dirty(DIRTY_RASTER_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_UVSCALEOFFSET);
	gstate_c.submitType = SubmitType::SPLINE;

	int bytesRead = 0;
	drawEngineCommon_->SubmitCurve(control_points, indices, surface, gstate.vertType, &bytesRead, "spline");

	gstate_c.Dirty(DIRTY_RASTER_STATE | DIRTY_VERTEXSHADER_STATE | DIRTY_UVSCALEOFFSET);
	gstate_c.submitType = SubmitType::DRAW;

	// After drawing, we advance pointers - see SubmitPrim which does the same.
	int count = surface.num_points_u * surface.num_points_v;
	gstate_c.AdvanceVerts(vertType, count, bytesRead);
}

void GPUCommonHWMetal::Execute_BlockTransferStart(u32 op, u32 diff) {
	drawEngineCommon_->FlushQueuedDepth();
	Flush();

	PROFILE_THIS_SCOPE("block");  // don't include the flush in the profile, would be misleading.

	gstate_c.framebufFormat = gstate.FrameBufFormat();

	// and take appropriate action. This is a block transfer between RAM and VRAM, or vice versa.
	// Can we skip this on SkipDraw?
	DoBlockTransfer(gstate_c.skipDrawReason);
}

void GPUCommonHWMetal::Execute_TexSize0(u32 op, u32 diff) {
	// Render to texture may have overridden the width/height.
	// Don't reset it unless the size is different / the texture has changed.
	if (diff || gstate_c.IsDirty(DIRTY_TEXTURE_IMAGE | DIRTY_TEXTURE_PARAMS)) {
		gstate_c.curTextureWidth = gstate.getTextureWidth(0);
		gstate_c.curTextureHeight = gstate.getTextureHeight(0);
		// We will need to reset the texture now.
		gstate_c.Dirty(DIRTY_TEXTURE_PARAMS);
	}
}

void GPUCommonHWMetal::Execute_TexLevel(u32 op, u32 diff) {
	// TODO: If you change the rules here, don't forget to update the inner interpreter in Execute_Prim.
	if (diff == 0xFFFFFFFF)
		return;

	gstate.texlevel ^= diff;

	if (diff & 0xFF0000) {
		// Piggyback on this flag for 3D textures.
		gstate_c.Dirty(DIRTY_MIPBIAS);
	}
	if (gstate.getTexLevelMode() != GE_TEXLEVEL_MODE_AUTO && (0x00FF0000 & gstate.texlevel) != 0) {
		Flush();
	}

	gstate.texlevel ^= diff;

	gstate_c.Dirty(DIRTY_TEXTURE_PARAMS | DIRTY_FRAGMENTSHADER_STATE);
}

void GPUCommonHWMetal::Execute_LoadClut(u32 op, u32 diff) {
	gstate_c.Dirty(DIRTY_TEXTURE_PARAMS);
	textureCache_->LoadClut(gstate.getClutAddress(), gstate.getClutLoadBytes(), &recorder_);
}

void GPUCommonHWMetal::Execute_WorldMtxNum(u32 op, u32 diff) {
	if (!currentList) {
		gstate.worldmtxnum = (GE_CMD_WORLDMATRIXNUMBER << 24) | (op & 0xF);
		return;
	}

	// This is almost always followed by GE_CMD_WORLDMATRIXDATA.
	const u32_le *src = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	u32 *dst = (u32 *)(gstate.worldMatrix + (op & 0xF));
	const int end = 12 - (op & 0xF);
	int i = 0;

	// We must record the individual data commands while debugRecording_.
	bool fastLoad = !debugRecording_ && end > 0;
	// Stalling in the middle of a matrix would be stupid, I doubt this check is necessary.
	if (currentList->pc < currentList->stall && currentList->pc + end * 4 >= currentList->stall) {
		fastLoad = false;
	}

	if (fastLoad) {
		while ((src[i] >> 24) == GE_CMD_WORLDMATRIXDATA) {
			const u32 newVal = src[i] << 8;
			if (dst[i] != newVal) {
				Flush();
				dst[i] = newVal;
				gstate_c.Dirty(DIRTY_WORLDMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX);
			}
			if (++i >= end) {
				break;
			}
		}
	}

	const int count = i;
	gstate.worldmtxnum = (GE_CMD_WORLDMATRIXNUMBER << 24) | ((op & 0xF) + count);

	// Skip over the loaded data, it's done now.
	currentList->pc += count * 4;
	UpdatePC(currentList->pc, currentList->pc);
}

void GPUCommonHWMetal::Execute_WorldMtxData(u32 op, u32 diff) {
	// Note: it's uncommon to get here now, see above.
	int num = gstate.worldmtxnum & 0x00FFFFFF;
	u32 newVal = op << 8;
	if (num < 12 && newVal != ((const u32 *)gstate.worldMatrix)[num]) {
		Flush();
		((u32 *)gstate.worldMatrix)[num] = newVal;
		gstate_c.Dirty(DIRTY_WORLDMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX);
	}
	num++;
	gstate.worldmtxnum = (GE_CMD_WORLDMATRIXNUMBER << 24) | (num & 0x00FFFFFF);
	gstate.worldmtxdata = GE_CMD_WORLDMATRIXDATA << 24;
}

void GPUCommonHWMetal::Execute_ViewMtxNum(u32 op, u32 diff) {
	if (!currentList) {
		gstate.viewmtxnum = (GE_CMD_VIEWMATRIXNUMBER << 24) | (op & 0xF);
		return;
	}

	// This is almost always followed by GE_CMD_VIEWMATRIXDATA.
	const u32_le *src = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	u32 *dst = (u32 *)(gstate.viewMatrix + (op & 0xF));
	const int end = 12 - (op & 0xF);
	int i = 0;

	bool fastLoad = !debugRecording_ && end > 0;
	if (currentList->pc < currentList->stall && currentList->pc + end * 4 >= currentList->stall) {
		fastLoad = false;
	}

	if (fastLoad) {
		while ((src[i] >> 24) == GE_CMD_VIEWMATRIXDATA) {
			const u32 newVal = src[i] << 8;
			if (dst[i] != newVal) {
				Flush();
				dst[i] = newVal;
				gstate_c.Dirty(DIRTY_VIEWMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX) ;
			}
			if (++i >= end) {
				break;
			}
		}
	}

	const int count = i;
	gstate.viewmtxnum = (GE_CMD_VIEWMATRIXNUMBER << 24) | ((op & 0xF) + count);

	// Skip over the loaded data, it's done now.
	currentList->pc += count * 4;
	UpdatePC(currentList->pc, currentList->pc);
}

void GPUCommonHWMetal::Execute_ViewMtxData(u32 op, u32 diff) {
	// Note: it's uncommon to get here now, see above.
	int num = gstate.viewmtxnum & 0x00FFFFFF;
	u32 newVal = op << 8;
	if (num < 12 && newVal != ((const u32 *)gstate.viewMatrix)[num]) {
		Flush();
		((u32 *)gstate.viewMatrix)[num] = newVal;
		gstate_c.Dirty(DIRTY_VIEWMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX);
	}
	num++;
	gstate.viewmtxnum = (GE_CMD_VIEWMATRIXNUMBER << 24) | (num & 0x00FFFFFF);
	gstate.viewmtxdata = GE_CMD_VIEWMATRIXDATA << 24;
}

void GPUCommonHWMetal::Execute_ProjMtxNum(u32 op, u32 diff) {
	if (!currentList) {
		gstate.projmtxnum = (GE_CMD_PROJMATRIXNUMBER << 24) | (op & 0xF);
		return;
	}

	// This is almost always followed by GE_CMD_PROJMATRIXDATA.
	const u32_le *src = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	u32 *dst = (u32 *)(gstate.projMatrix + (op & 0xF));
	const int end = 16 - (op & 0xF);
	int i = 0;

	bool fastLoad = !debugRecording_;
	if (currentList->pc < currentList->stall && currentList->pc + end * 4 >= currentList->stall) {
		fastLoad = false;
	}

	if (fastLoad) {
		while ((src[i] >> 24) == GE_CMD_PROJMATRIXDATA) {
			const u32 newVal = src[i] << 8;
			if (dst[i] != newVal) {
				Flush();
				dst[i] = newVal;
				gstate_c.Dirty(DIRTY_PROJMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX);
			}
			if (++i >= end) {
				break;
			}
		}
	}

	const int count = i;
	gstate.projmtxnum = (GE_CMD_PROJMATRIXNUMBER << 24) | ((op & 0xF) + count);

	// Skip over the loaded data, it's done now.
	currentList->pc += count * 4;
	UpdatePC(currentList->pc, currentList->pc);
}

void GPUCommonHWMetal::Execute_ProjMtxData(u32 op, u32 diff) {
	// Note: it's uncommon to get here now, see above.
	int num = gstate.projmtxnum & 0x00FFFFFF;
	u32 newVal = op << 8;
	if (num < 16 && newVal != ((const u32 *)gstate.projMatrix)[num]) {
		Flush();
		((u32 *)gstate.projMatrix)[num] = newVal;
		gstate_c.Dirty(DIRTY_PROJMATRIX | DIRTY_WORLD_VIEW_PROJ_MATRIX | DIRTY_VIEW_PROJ_MATRIX | DIRTY_CULL_MATRIX);
	}
	num++;
	if (num <= 16)
		gstate.projmtxnum = (GE_CMD_PROJMATRIXNUMBER << 24) | (num & 0x00FFFFFF);
	gstate.projmtxdata = GE_CMD_PROJMATRIXDATA << 24;
}

void GPUCommonHWMetal::Execute_TgenMtxNum(u32 op, u32 diff) {
	if (!currentList) {
		gstate.texmtxnum = (GE_CMD_TGENMATRIXNUMBER << 24) | (op & 0xF);
		return;
	}

	// This is almost always followed by GE_CMD_TGENMATRIXDATA.
	const u32_le *src = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	u32 *dst = (u32 *)(gstate.tgenMatrix + (op & 0xF));
	const int end = 12 - (op & 0xF);
	int i = 0;

	bool fastLoad = !debugRecording_ && end > 0;
	if (currentList->pc < currentList->stall && currentList->pc + end * 4 >= currentList->stall) {
		fastLoad = false;
	}

	if (fastLoad) {
		while ((src[i] >> 24) == GE_CMD_TGENMATRIXDATA) {
			const u32 newVal = src[i] << 8;
			if (dst[i] != newVal) {
				Flush();
				dst[i] = newVal;
				// We check the matrix to see if we need projection.
				gstate_c.Dirty(DIRTY_TEXMATRIX | DIRTY_FRAGMENTSHADER_STATE);
			}
			if (++i >= end) {
				break;
			}
		}
	}

	const int count = i;
	gstate.texmtxnum = (GE_CMD_TGENMATRIXNUMBER << 24) | ((op & 0xF) + count);

	// Skip over the loaded data, it's done now.
	currentList->pc += count * 4;
	UpdatePC(currentList->pc, currentList->pc);
}

void GPUCommonHWMetal::Execute_TgenMtxData(u32 op, u32 diff) {
	// Note: it's uncommon to get here now, see above.
	int num = gstate.texmtxnum & 0x00FFFFFF;
	u32 newVal = op << 8;
	if (num < 12 && newVal != ((const u32 *)gstate.tgenMatrix)[num]) {
		Flush();
		((u32 *)gstate.tgenMatrix)[num] = newVal;
		gstate_c.Dirty(DIRTY_TEXMATRIX | DIRTY_FRAGMENTSHADER_STATE);  // We check the matrix to see if we need projection
	}
	num++;
	gstate.texmtxnum = (GE_CMD_TGENMATRIXNUMBER << 24) | (num & 0x00FFFFFF);
	gstate.texmtxdata = GE_CMD_TGENMATRIXDATA << 24;
}

void GPUCommonHWMetal::Execute_BoneMtxNum(u32 op, u32 diff) {
	if (!currentList) {
		gstate.boneMatrixNumber = (GE_CMD_BONEMATRIXNUMBER << 24) | (op & 0x7F);
		return;
	}

	// This is almost always followed by GE_CMD_BONEMATRIXDATA.
	const u32_le *src = (const u32_le *)Memory::GetPointerUnchecked(currentList->pc + 4);
	u32 *dst = (u32 *)(gstate.boneMatrix + (op & 0x7F));
	const int end = 12 * 8 - (op & 0x7F);
	int i = 0;

	bool fastLoad = !debugRecording_ && end > 0;
	if (currentList->pc < currentList->stall && currentList->pc + end * 4 >= currentList->stall) {
		fastLoad = false;
	}

	if (fastLoad) {
		while ((src[i] >> 24) == GE_CMD_BONEMATRIXDATA) {
			dst[i] = src[i] << 8;
			if (++i >= end) {
				break;
			}
		}
	}

	const int count = i;
	gstate.boneMatrixNumber = (GE_CMD_BONEMATRIXNUMBER << 24) | ((op & 0x7F) + count);

	// Skip over the loaded data, it's done now.
	currentList->pc += count * 4;
	UpdatePC(currentList->pc, currentList->pc);
}

void GPUCommonHWMetal::Execute_BoneMtxData(u32 op, u32 diff) {
	// Note: it's uncommon to get here now, see above.
	int num = gstate.boneMatrixNumber & 0x00FFFFFF;
	u32 newVal = op << 8;
	if (num < 96 && newVal != ((const u32 *)gstate.boneMatrix)[num]) {
		// Bone matrices should NOT flush, as we're always doing skinning in decode nowadays!
		((u32 *)gstate.boneMatrix)[num] = newVal;
	}
	num++;
	gstate.boneMatrixNumber = (GE_CMD_BONEMATRIXNUMBER << 24) | (num & 0x00FFFFFF);
	gstate.boneMatrixData = GE_CMD_BONEMATRIXDATA << 24;
}

void GPUCommonHWMetal::Execute_TexFlush(u32 op, u32 diff) {
	// Games call this when they need the effect of drawing to be visible to texturing.
	// And for a bunch of other reasons, but either way, this is what we need to do.
	// It's possible we could also use this as a hint for the texture cache somehow.
	framebufferManager_->DiscardFramebufferCopy();
}

u32 GPUCommonHWMetal::DrawSync(int mode) {
	drawEngineCommon_->FlushQueuedDepth();
	return GPUCommon::DrawSync(mode);
}

int GPUCommonHWMetal::ListSync(int listid, int mode) {
	drawEngineCommon_->FlushQueuedDepth();
	return GPUCommon::ListSync(listid, mode);
}

void GPUCommonHWMetal::FormatGPUStatsCommon(StringWriter &w) {
	float vertexAverageCycles = gpuStats.perFrame.numVertsSubmitted > 0 ? (float)gpuStats.perFrame.vertexGPUCycles / (float)gpuStats.perFrame.numVertsSubmitted : 0.0f;
	w.F("DL processing time: %0.2f ms\n", gpuStats.perFrame.msProcessingDisplayLists * 1000.0f);
	w.F("%d enqueue, %d updstall, %d drawsync, %d listsync, %d geint (td: %d)\n",
		gpuStats.perFrame.numEnqueue,
		gpuStats.perFrame.numUpdateStall,
		gpuStats.perFrame.numDrawSyncs,
		gpuStats.perFrame.numListSyncs,
		gpuStats.perFrame.numGEInterrupts,
		gstate_c.textureSyncTimeDomain);
	w.F("Draw: %d (%d dec, %d culled), flushes %d, clears %d, bbox jumps %d\n"
		"%d soft. Vertices: %d dec: %d drawn: %d clipped tris: %d\n"
		"GPU cycles: %d (%0.1f per vertex)\n",
		gpuStats.perFrame.numDrawCalls,
		gpuStats.perFrame.numVertexDecodes,
		gpuStats.perFrame.numCulledDraws,
		gpuStats.perFrame.numFlushes,
		gpuStats.perFrame.numClears,
		gpuStats.perFrame.numBBOXJumps,
		gpuStats.perFrame.numSoftTransformedDraws,
		gpuStats.perFrame.numVertsSubmitted,
		gpuStats.perFrame.numVertsDecoded,
		gpuStats.perFrame.numVertsDrawn,
		gpuStats.perFrame.numSoftClippedTriangles,
		gpuStats.perFrame.vertexGPUCycles + gpuStats.perFrame.otherGPUCycles,
		vertexAverageCycles);
	w.F("FBOs active: %d (evaluations: %d, created %d)\n"
		"Text: %d (s: %d), dec: %d, invalidated: %d, changed %d, hashed: %d kB, clut %d\n"
		"readbacks %d (%d non-block), upload %d (cached %d), depal %d, video %d\n"
		"block transfers: %d\n"
		"Cpy: depth %d, color %d, reint %d, blend %d, self %d\n",
		(int)framebufferManager_->NumVFBs(),
		gpuStats.perFrame.numFramebufferEvaluations,
		gpuStats.perFrame.numFBOsCreated,
		(int)textureCache_->NumLoadedTextures(),
		(int)textureCache_->NumSecondaryTextures(),
		gpuStats.perFrame.numTexturesDecoded,
		gpuStats.perFrame.numTextureInvalidations,
		gpuStats.perFrame.numTexturesChanged,
		gpuStats.perFrame.numTextureDataBytesHashed / 1024,
		gpuStats.perFrame.numClutTextures,
		gpuStats.perFrame.numBlockingReadbacks,
		gpuStats.perFrame.numReadbacks,
		gpuStats.perFrame.numUploads,
		gpuStats.perFrame.numCachedUploads,
		gpuStats.perFrame.numDepal,
		gpuStats.perFrame.numVideoTextures,
		gpuStats.perFrame.numBlockTransfers,
		gpuStats.perFrame.numDepthCopies,
		gpuStats.perFrame.numColorCopies,
		gpuStats.perFrame.numReinterpretCopies,
		gpuStats.perFrame.numCopiesForShaderBlend,
		gpuStats.perFrame.numCopiesForSelfTex);

	if (gpuStats.perFrame.numReplacerTrackedTex) {
		w.F("replacer: tracks %d references, %d unique textures\n",
			gpuStats.perFrame.numReplacerTrackedTex,
			gpuStats.perFrame.numCachedReplacedTextures);
	}

	if (PSP_CoreParameter().compat.flags().SoftwareRasterDepth) {
		w.F("Z-rast: %0.2f+%0.2f+%0.2f (total %0.2f/%0.2f) ms\n"
			"Z-rast: %d prim, %d nopix, %d small, %d earlysize, %d zcull, %d box\n%s",
			gpuStats.perFrame.msPrepareDepth * 1000.0,
			gpuStats.perFrame.msCullDepth * 1000.0,
			gpuStats.perFrame.msRasterizeDepth * 1000.0,
			(gpuStats.perFrame.msPrepareDepth + gpuStats.perFrame.msCullDepth + gpuStats.perFrame.msRasterizeDepth) * 1000.0,
			gpuStats.perFrame.msRasterTimeAvailable * 1000.0,
			gpuStats.perFrame.numDepthRasterPrims,
			gpuStats.perFrame.numDepthRasterNoPixels,
			gpuStats.perFrame.numDepthRasterTooSmall,
			gpuStats.perFrame.numDepthRasterEarlySize,
			gpuStats.perFrame.numDepthRasterZCulled,
			gpuStats.perFrame.numDepthEarlyBoxCulled,
			debugRecording_ ? "(debug-recording)" : ""
		);
	}
}


// One row of a block transfer whose source and destination overlap, as the GE copies it (ppsspp-re geprobe
// exp219, exp220: 562 of 570 cases exact). The source is read in whole aligned 16-byte blocks, four at a
// time from the row's first block, and each byte placed at its destination address in a buffer of four
// 16-byte slots (indexed by destination address bits 4-5, tagged by destination block), only the bytes
// inside the transfer marked written. Placing into a slot tagged with another block writes that block's
// written bytes to memory and retags the slot, keeping its bytes. Reading a source block that its slot is
// tagged with takes the slot's bytes as they are, stale ones included. The slots are written out after the row.
static void BlockTransferOverlappingRow(u32 srcAddr, u32 dstAddr, u32 bytes) {
	struct Slot {
		u32 block;
		bool used;
		u16 written;
		u8 data[16];
	};
	Slot slots[4]{};
	auto writeOut = [](const Slot &slot) {
		if (!slot.used || !slot.written)
			return;
		u8 *p = Memory::GetPointerWriteUnchecked(slot.block << 4);
		for (int k = 0; k < 16; ++k) {
			if (slot.written & (1 << k))
				p[k] = slot.data[k];
		}
	};
	const u32 firstBlock = srcAddr >> 4, lastBlock = (srcAddr + bytes - 1) >> 4;
	const u32 delta = dstAddr - srcAddr;
	for (u32 group = firstBlock; group <= lastBlock; group += 4) {
		const u32 count = std::min(4u, lastBlock - group + 1);
		u8 read[4][16];
		for (u32 i = 0; i < count; ++i) {
			const u32 block = group + i;
			const Slot &slot = slots[block & 3];
			if (slot.used && slot.block == block && slot.written)
				memcpy(read[i], slot.data, 16);
			else
				memcpy(read[i], Memory::GetPointerUnchecked(block << 4), 16);
		}
		for (u32 i = 0; i < count; ++i) {
			for (u32 k = 0; k < 16; ++k) {
				const u32 addr = ((group + i) << 4) + k;
				const u32 dst = addr + delta;
				Slot &slot = slots[(dst >> 4) & 3];
				if (!slot.used || slot.block != dst >> 4) {
					writeOut(slot);
					slot.used = true;
					slot.block = dst >> 4;
					slot.written = 0;
				}
				slot.data[dst & 15] = read[i][k];
				if (addr - srcAddr < bytes)
					slot.written |= 1 << (dst & 15);
			}
		}
	}
	for (const Slot &slot : slots)
		writeOut(slot);
}


void GPUCommonHWMetal::BeginHostFrame(const DisplayLayoutConfig &config) {
	ReapplyGfxState();

	// TODO: Assume config may have changed - maybe move to resize.
	gstate_c.Dirty(DIRTY_ALL);

	UpdateCmdInfo();

	UpdateMSAALevel(draw_);
	CheckConfigChanged(config);
	CheckDisplayResized();
	CheckRenderResized(config);

	// After the resizes, which ask for the post shaders to be rebuilt at the new size.
	if (framebufferManager_) {
		framebufferManager_->CheckPostShaders(config);
	}
}

void GPUCommonHWMetal::Reinitialize() {
	memset(dls, 0, sizeof(dls));
	for (int i = 0; i < DisplayListMaxCount; ++i) {
		dls[i].state = PSP_GE_DL_STATE_NONE;
		dls[i].waitUntilTicks = 0;
	}
	// This is what sceKernelLoadExec gets, and whatever the old executable still had queued goes with
	// it.  Left behind, those ids would now be lists with no state and a pc of 0, waiting their turn
	// behind the first list of the new executable.  Crazy Taxi: Fare Wars starts its games this way, #19894.
	dlQueue.clear();

	// The GE driver starts over as well, and the first thing it does is run a list that sets every
	// register to zero, and all the matrices.  So a program started this way finds the GE exactly as
	// one booted directly does, not as the previous one left it.
	gstate.Reset();
	gstate_c.offsetAddr = 0;
	gstate_c.vertexAddr = 0;
	gstate_c.indexAddr = 0;
	ResetMatrices();
	gstate_c.Dirty(DIRTY_ALL);

	nextListID = 0;
	currentList = nullptr;
	interruptRunning = false;
	isbreak = false;
	drawCompleteTicks = 0;
	busyTicks = 0;
	interruptsEnabled_ = true;
	videos_.clear();

	if (textureCache_)
		textureCache_->Clear(true);
	if (framebufferManager_)
		framebufferManager_->DestroyAllFBOs();
}

void GPUCommonHWMetal::Execute_End(u32 op, u32 diff) {
	if (flushOnParams_) {
		drawEngineCommon_->FlushQueuedDepth();
		Flush();
	}

	const u32 prev = Memory::ReadUnchecked_U32(currentList->pc - 4);
	UpdatePC(currentList->pc, currentList->pc);
	// Count in a few extra cycles on END.
	cyclesExecuted += 60;

	switch (prev >> 24) {
	case GE_CMD_SIGNAL:
		{
			// TODO: see http://code.google.com/p/jpcsp/source/detail?r=2935#
			SignalBehavior behaviour = static_cast<SignalBehavior>((prev >> 16) & 0xFF);
			const int signal = prev & 0xFFFF;
			const int enddata = op & 0xFFFF;
			bool trigger = true;
			currentList->subIntrToken = signal;

			switch (behaviour) {
			case PSP_GE_SIGNAL_HANDLER_SUSPEND:
				// Suspend the list, and call the signal handler.  When it's done, resume.
				// Before sdkver 0x02000010, listsync should return paused.
				if (sceKernelGetCompiledSdkVersion() <= 0x02000010)
					currentList->state = PSP_GE_DL_STATE_PAUSED;
				currentList->signal = behaviour;
				DEBUG_LOG(Log::G3D, "Signal with wait. signal/end: %04x %04x", signal, enddata);
				break;
			case PSP_GE_SIGNAL_HANDLER_CONTINUE:
				// Resume the list right away, then call the handler.
				currentList->signal = behaviour;
				DEBUG_LOG(Log::G3D, "Signal without wait. signal/end: %04x %04x", signal, enddata);
				break;
			case PSP_GE_SIGNAL_HANDLER_PAUSE:
				// Pause the list instead of ending at the next FINISH.
				// Call the handler with the PAUSE signal value at that FINISH.
				// Technically, this ought to trigger an interrupt, but it won't do anything.
				// But right now, signal is always reset by interrupts, so that causes pause to not work.
				trigger = false;
				currentList->signal = behaviour;
				// The list reads as paused from here on, though it keeps executing until that FINISH.
				// If it stalls before it, it's stuck: only a running list's stall address is updated.
				currentList->state = PSP_GE_DL_STATE_PAUSED;
				DEBUG_LOG(Log::G3D, "Signal with Pause. signal/end: %04x %04x", signal, enddata);
				break;
			case PSP_GE_SIGNAL_SYNC:
				// Acts as a memory barrier, never calls any user code.
				// Technically, this ought to trigger an interrupt, but it won't do anything.
				// Triggering here can cause incorrect rescheduling, which breaks 3rd Birthday.
				// However, this is likely a bug in how GE signal interrupts are handled.
				trigger = false;
				currentList->signal = behaviour;
				DEBUG_LOG(Log::G3D, "Signal with Sync. signal/end: %04x %04x", signal, enddata);
				break;
			case PSP_GE_SIGNAL_JUMP:
			case PSP_GE_SIGNAL_RJUMP:
			case PSP_GE_SIGNAL_OJUMP:
				{
					trigger = false;
					currentList->signal = behaviour;
					// pc will be increased after we return, counteract that.
					u32 target = (((signal << 16) | enddata) & 0xFFFFFFFC) - 4;
					const char *targetType = "absolute";
					if (behaviour == PSP_GE_SIGNAL_RJUMP) {
						target += currentList->pc - 4;
						targetType = "relative";
					} else if (behaviour == PSP_GE_SIGNAL_OJUMP) {
						target = gstate_c.getRelativeAddress(target);
						targetType = "origin";
					}

					if (!Memory::IsValidAddress(target)) {
						ERROR_LOG_REPORT(Log::G3D, "Signal with Jump (%s): bad address. signal/end: %04x %04x", targetType, signal, enddata);
						UpdateState(GPUSTATE_ERROR);
					} else {
						UpdatePC(currentList->pc, target);
						currentList->pc = target;
						DEBUG_LOG(Log::G3D, "Signal with Jump (%s). signal/end: %04x %04x", targetType, signal, enddata);
					}
				}
				break;
			case PSP_GE_SIGNAL_CALL:
			case PSP_GE_SIGNAL_RCALL:
			case PSP_GE_SIGNAL_OCALL:
				{
					trigger = false;
					currentList->signal = behaviour;
					// pc will be increased after we return, counteract that.
					u32 target = (((signal << 16) | enddata) & 0xFFFFFFFC) - 4;
					const char *targetType = "absolute";
					if (behaviour == PSP_GE_SIGNAL_RCALL) {
						target += currentList->pc - 4;
						targetType = "relative";
					} else if (behaviour == PSP_GE_SIGNAL_OCALL) {
						target = gstate_c.getRelativeAddress(target);
						targetType = "origin";
					}

					if (currentList->stackptr == ARRAY_SIZE(currentList->stack)) {
						ERROR_LOG_REPORT(Log::G3D, "Signal with Call (%s): stack full. signal/end: %04x %04x", targetType, signal, enddata);
					} else if (!Memory::IsValidAddress(target)) {
						ERROR_LOG_REPORT(Log::G3D, "Signal with Call (%s): bad address. signal/end: %04x %04x", targetType, signal, enddata);
						UpdateState(GPUSTATE_ERROR);
					} else {
						// TODO: This might save/restore other state...
						auto &stackEntry = currentList->stack[currentList->stackptr++];
						stackEntry.pc = currentList->pc;
						stackEntry.offsetAddr = gstate_c.offsetAddr;
						stackEntry.baseAddr = gstate.base;
						UpdatePC(currentList->pc, target);
						currentList->pc = target;
						DEBUG_LOG(Log::G3D, "Signal with Call (%s). signal/end: %04x %04x", targetType, signal, enddata);
					}
				}
				break;
			case PSP_GE_SIGNAL_RET:
				{
					trigger = false;
					currentList->signal = behaviour;
					if (currentList->stackptr == 0) {
						ERROR_LOG_REPORT(Log::G3D, "Signal with Return: stack empty. signal/end: %04x %04x", signal, enddata);
					} else {
						// TODO: This might save/restore other state...
						auto &stackEntry = currentList->stack[--currentList->stackptr];
						gstate_c.offsetAddr = stackEntry.offsetAddr;
						gstate.base = stackEntry.baseAddr;
						UpdatePC(currentList->pc, stackEntry.pc);
						currentList->pc = stackEntry.pc;
						DEBUG_LOG(Log::G3D, "Signal with Return. signal/end: %04x %04x", signal, enddata);
					}
				}
				break;
			default:
				ERROR_LOG_REPORT(Log::G3D, "UNKNOWN Signal UNIMPLEMENTED %i ! signal/end: %04x %04x", behaviour, signal, enddata);
				break;
			}
			// TODO: Technically, jump/call/ret should generate an interrupt, but before the pc change maybe?
			if (currentList->interruptsEnabled && trigger) {
				if (__GeTriggerInterrupt(currentList->id, currentList->pc, startingTicks + cyclesExecuted)) {
					currentList->pendingInterrupt = true;
					UpdateState(GPUSTATE_INTERRUPT);
				}
			}
		}
		break;
	case GE_CMD_FINISH:
		switch (currentList->signal) {
		case PSP_GE_SIGNAL_HANDLER_PAUSE:
			currentList->state = PSP_GE_DL_STATE_PAUSED;
			if (currentList->interruptsEnabled) {
				if (__GeTriggerInterrupt(currentList->id, currentList->pc, startingTicks + cyclesExecuted)) {
					currentList->pendingInterrupt = true;
					UpdateState(GPUSTATE_INTERRUPT);
				}
			}
			break;

		case PSP_GE_SIGNAL_SYNC:
			currentList->signal = PSP_GE_SIGNAL_NONE;
			// TODO: Technically this should still cause an interrupt.  Probably for memory sync.
			break;

		default:
			FlushImm();
			currentList->subIntrToken = prev & 0xFFFF;
			UpdateState(GPUSTATE_DONE);

			if (currentList->interruptsEnabled && __GeTriggerInterrupt(currentList->id, currentList->pc, startingTicks + cyclesExecuted)) {
				// The context is restored once the finish callback has run, which gets to see the state
				// the list left behind.  Nothing else runs before then, see InterruptEnd().
				currentList->pendingInterrupt = true;
			} else {
				// No interrupt to wait for, so this is it.
				if (currentList->started && currentList->context.IsValid()) {
					gstate.Restore(currentList->context);
					ReapplyGfxState();
					// Don't restore the context again.
					currentList->started = false;
				}
				currentList->state = PSP_GE_DL_STATE_COMPLETED;
				currentList->waitUntilTicks = startingTicks + cyclesExecuted;
				busyTicks = std::max(busyTicks, currentList->waitUntilTicks);
				__GeTriggerSync(GPU_SYNC_LIST, currentList->id, currentList->waitUntilTicks);
			}
			break;
		}
		break;
	default:
		DEBUG_LOG(Log::G3D, "END: Not finished: %06x", prev & 0xFFFFFF);
		break;
	}
}

void GPUCommonHWMetal::Execute_ImmVertexAlphaPrim(u32 op, u32 diff) {
	// Safety check.
	if (immCount_ >= MAX_IMMBUFFER_SIZE) {
		// Only print once for each overrun.
		if (immCount_ == MAX_IMMBUFFER_SIZE) {
			ERROR_LOG_REPORT_ONCE(exceed_imm_buffer, Log::G3D, "Exceeded immediate draw buffer size. gstate.imm_ap=%06x , prim=%d", gstate.imm_ap & 0xFFFFFF, (int)immPrim_);
		}
		if (immCount_ < 0x7fffffff)  // Paranoia :)
			immCount_++;
		return;
	}

	const int prim = (op >> 8) & 0x7;
	if (prim != GE_PRIM_KEEP_PREVIOUS) {
		// Flush before changing the prim type.  Only continue can be used to continue a prim.
		FlushImm();
	}

	TransformedVertex &v = immBuffer_[immCount_++];

	// ThrillVille does a clear with this, additional parameters found via tests.
	// The current vtype affects how the coordinate is processed.
	if (gstate.isModeThrough()) {
		v.x = ((int)(gstate.imm_vscx & 0xFFFF) - 0x8000) / 16.0f;
		v.y = ((int)(gstate.imm_vscy & 0xFFFF) - 0x8000) / 16.0f;
	} else {
		int offsetX = gstate.getOffsetX16();
		int offsetY = gstate.getOffsetY16();
		v.x = ((int)(gstate.imm_vscx & 0xFFFF) - offsetX) / 16.0f;
		v.y = ((int)(gstate.imm_vscy & 0xFFFF) - offsetY) / 16.0f;
	}
	v.z = gstate.imm_vscz & 0xFFFF;
	v.pos_w = 1.0f;
	v.u = getFloat24(gstate.imm_vtcs);
	v.v = getFloat24(gstate.imm_vtct);
	v.uv_w = getFloat24(gstate.imm_vtcq);
	v.color0_32 = (gstate.imm_cv & 0xFFFFFF) | (gstate.imm_ap << 24);
	// TODO: When !gstate.isModeThrough(), direct fog coefficient (0 = entirely fog), ignore fog flag (also GE_IMM_FOG.)
	v.fog = (gstate.imm_fc & 0xFF) / 255.0f;
	// TODO: Apply if gstate.isUsingSecondaryColor() && !gstate.isModeThrough(), ignore lighting flag.
	v.color1_32 = gstate.imm_scv & 0xFFFFFF;
	if (prim != GE_PRIM_KEEP_PREVIOUS) {
		immPrim_ = (GEPrimitiveType)prim;
		// Flags seem to only be respected from the first prim.
		immFlags_ = op & 0x00FFF800;
		immFirstSent_ = false;
	} else if (prim == GE_PRIM_KEEP_PREVIOUS && immPrim_ != GE_PRIM_INVALID) {
		static constexpr int flushPrimCount[] = { 1, 2, 0, 3, 0, 0, 2, 0 };
		// Instead of finding a proper point to flush, we just emit prims when we can.
		if (immCount_ == flushPrimCount[immPrim_ & 7])
			FlushImm();
	} else {
		ERROR_LOG_REPORT_ONCE(imm_draw_prim, Log::G3D, "Immediate draw: Unexpected primitive %d at count %d", prim, immCount_);
	}
}

void GPUCommonHWMetal::FlushImm() {
	if (immCount_ == 0 || immPrim_ == GE_PRIM_INVALID)
		return;

	if (gstate_c.skipDrawReason & (SKIPDRAW_SKIPFRAME | SKIPDRAW_NON_DISPLAYED_FB)) {
		// No idea how many cycles to skip, heh.
		immCount_ = 0;
		return;
	}

	// Ignore immediate point primitives with a X/Y of 0.
	// These are accidentally created when games clear the graphics state.
	if (immCount_ == 1 && immPrim_ == GE_PRIM_POINTS && immBuffer_[0].x == 0 && immBuffer_[0].y == 0 && immBuffer_[0].z == 0 && immBuffer_[0].color0_32 == 0) {
		immCount_ = 0;
		return;
	}

	SetDrawType(DRAW_PRIM, immPrim_);
	gstate_c.framebufFormat = gstate.FrameBufFormat();

	VirtualFramebuffer *vfb = nullptr;
	if (framebufferManager_) {
		bool changed;
		vfb = framebufferManager_->SetRenderFrameBuffer(gstate_c.IsDirty(DIRTY_FRAMEBUF), gstate_c.skipDrawReason, &changed);
	}
	if (vfb) {
		CheckDepthUsage(vfb);
	}

	bool antialias = (immFlags_ & GE_IMM_ANTIALIAS) != 0;
	bool prevAntialias = gstate.isAntiAliasEnabled();
	bool shading = (immFlags_ & GE_IMM_SHADING) != 0;
	bool prevShading = gstate.getShadeMode() == GE_SHADE_GOURAUD;
	bool cullEnable = (immFlags_ & GE_IMM_CULLENABLE) != 0;
	bool prevCullEnable = gstate.isCullEnabled();
	int cullMode = (immFlags_ & GE_IMM_CULLFACE) != 0 ? 1 : 0;
	bool texturing = (immFlags_ & GE_IMM_TEXTURE) != 0;
	bool prevTexturing = gstate.isTextureMapEnabled();
	bool fog = (immFlags_ & GE_IMM_FOG) != 0;
	bool prevFog = gstate.isFogEnabled();
	bool dither = (immFlags_ & GE_IMM_DITHER) != 0;
	bool prevDither = gstate.isDitherEnabled();

	if ((immFlags_ & GE_IMM_CLIPMASK) != 0) {
		WARN_LOG_REPORT_ONCE(geimmclipvalue, Log::G3D, "Imm vertex used clip value, flags=%06x", immFlags_);
	}

	bool changed = texturing != prevTexturing || cullEnable != prevCullEnable || dither != prevDither;
	changed = changed || prevShading != shading || prevFog != fog;
	// Always flush, even if the flags match: DispatchSubmitImm switches to through mode and a different
	// vertex decoder, which would otherwise apply to the draws already queued.
	Flush();
	if (changed) {
		gstate.antiAliasEnable = (GE_CMD_ANTIALIASENABLE << 24) | (int)antialias;
		gstate.shademodel = (GE_CMD_SHADEMODE << 24) | (int)shading;
		gstate.cullfaceEnable = (GE_CMD_CULLFACEENABLE << 24) | (int)cullEnable;
		gstate.textureMapEnable = (GE_CMD_TEXTUREMAPENABLE << 24) | (int)texturing;
		gstate.fogEnable = (GE_CMD_FOGENABLE << 24) | (int)fog;
		gstate.ditherEnable = (GE_CMD_DITHERENABLE << 24) | (int)dither;
		gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_RASTER_STATE | DIRTY_VIEWPORTSCISSOR_STATE | DIRTY_UVSCALEOFFSET);
	}

	drawEngineCommon_->DispatchSubmitImm(immPrim_, immBuffer_, immCount_, cullMode, immFirstSent_);
	immCount_ = 0;
	immFirstSent_ = true;

	if (changed) {
		Flush();
		gstate.antiAliasEnable = (GE_CMD_ANTIALIASENABLE << 24) | (int)prevAntialias;
		gstate.shademodel = (GE_CMD_SHADEMODE << 24) | (int)prevShading;
		gstate.cullfaceEnable = (GE_CMD_CULLFACEENABLE << 24) | (int)prevCullEnable;
		gstate.textureMapEnable = (GE_CMD_TEXTUREMAPENABLE << 24) | (int)prevTexturing;
		gstate.fogEnable = (GE_CMD_FOGENABLE << 24) | (int)prevFog;
		gstate.ditherEnable = (GE_CMD_DITHERENABLE << 24) | (int)prevDither;
		gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_RASTER_STATE | DIRTY_VIEWPORTSCISSOR_STATE | DIRTY_UVSCALEOFFSET);
	}
}

void GPUCommonHWMetal::DoBlockTransfer(u32 skipDrawReason) {
	u32 srcBasePtr = gstate.getTransferSrcAddress();
	u32 srcStride = gstate.getTransferSrcStride();

	u32 dstBasePtr = gstate.getTransferDstAddress();
	u32 dstStride = gstate.getTransferDstStride();

	int srcX = gstate.getTransferSrcX();
	int srcY = gstate.getTransferSrcY();

	int dstX = gstate.getTransferDstX();
	int dstY = gstate.getTransferDstY();

	int width = gstate.getTransferWidth();
	int height = gstate.getTransferHeight();

	int bpp = gstate.getTransferBpp();

	DEBUG_LOG(Log::G3D, "Block transfer: %08x/%x -> %08x/%x, %ix%ix%i (%i,%i)->(%i,%i)", srcBasePtr, srcStride, dstBasePtr, dstStride, width, height, bpp, srcX, srcY, dstX, dstY);
	gpuStats.perFrame.numBlockTransfers++;

	// For VRAM, we wrap around when outside valid memory (mirrors still work.)
	if ((srcBasePtr & 0x04800000) == 0x04800000)
		srcBasePtr &= ~0x00800000;
	if ((dstBasePtr & 0x04800000) == 0x04800000)
		dstBasePtr &= ~0x00800000;

	// Use height less one to account for width, which can be greater or less than stride, and then add it on for the last line.
	// NOTE: The sizes are only used for validity checks and memory info tracking.
	const uint32_t src = srcBasePtr + (srcY * srcStride + srcX) * bpp;
	const uint32_t dst = dstBasePtr + (dstY * dstStride + dstX) * bpp;
	const uint32_t srcSize = ((height - 1) * srcStride) + width * bpp;
	const uint32_t dstSize = ((height - 1) * dstStride) + width * bpp;

	bool srcDstOverlap = src + srcSize > dst && dst + dstSize > src;
	bool srcValid = Memory::IsValidRange(src, srcSize);
	bool dstValid = Memory::IsValidRange(dst, dstSize);
	bool srcWraps = Memory::IsVRAMAddress(srcBasePtr) && !srcValid;
	bool dstWraps = Memory::IsVRAMAddress(dstBasePtr) && !dstValid;

	char tag[128];
	size_t tagSize = 0;

	// Tell the framebuffer manager to take action if possible. If it does the entire thing, let's just return.
	if (!framebufferManager_ || !framebufferManager_->NotifyBlockTransferBefore(dstBasePtr, dstStride, dstX, dstY, srcBasePtr, srcStride, srcX, srcY, width, height, bpp, skipDrawReason)) {
		// Do the copy! (Hm, if we detect a drawn video frame (see below) then we could maybe skip this?)
		// Can use GetPointerUnchecked because we checked the addresses above. We could also avoid them
		// entirely by walking a couple of pointers...

		if (CopyBlockTransfer(srcBasePtr, srcStride, srcX, srcY, dstBasePtr, dstStride, dstX, dstY, width, height, bpp)) {
			// The backend copied it.
		} else if (srcStride == dstStride && (u32)width == srcStride && !srcDstOverlap && srcValid && dstValid) {
			// Simple case: just a straight copy, no overlap or wrapping.
			u32 srcLineStartAddr = srcBasePtr + (srcY * srcStride + srcX) * bpp;
			u32 dstLineStartAddr = dstBasePtr + (dstY * dstStride + dstX) * bpp;
			u32 bytesToCopy = width * height * bpp;

			const u8 *srcp = Memory::GetPointerOrException(srcLineStartAddr);
			u8 *dstp = Memory::GetPointerWriteOrException(dstLineStartAddr);
			memcpy(dstp, srcp, bytesToCopy);

			if (MemBlockInfoDetailed(bytesToCopy)) {
				NotifyMemInfoCopy(dst, src, bytesToCopy, "GPUBlockTransfer/");
			}
		} else if ((srcDstOverlap || srcWraps || dstWraps) && (srcValid || srcWraps) && (dstValid || dstWraps)) {
			// This path means we have either src/dst overlap, OR one or both of src and dst wrap.
			// This should be uncommon so it's the slowest path.
			u32 bytesToCopy = width * bpp;
			bool notifyDetail = MemBlockInfoDetailed(srcWraps || dstWraps ? 64 : bytesToCopy);
			bool notifyAll = !notifyDetail && MemBlockInfoDetailed(srcSize, dstSize);
			if (notifyDetail || notifyAll) {
				tagSize = FormatMemWriteTagAt(tag, sizeof(tag), "GPUBlockTransfer/", src, srcSize);
			}

			auto notifyingMemmove = [&](u32 d, u32 s, u32 sz) {
				const u8 *srcp = Memory::GetPointerOrException(s);
				u8 *dstp = Memory::GetPointerWriteOrException(d);
				memmove(dstp, srcp, sz);

				if (notifyDetail) {
					NotifyMemInfo(MemBlockFlags::READ, s, sz, tag, tagSize);
					NotifyMemInfo(MemBlockFlags::WRITE, d, sz, tag, tagSize);
				}
			};

			for (int y = 0; y < height; y++) {
				u32 srcLineStartAddr = srcBasePtr + ((y + srcY) * srcStride + srcX) * bpp;
				u32 dstLineStartAddr = dstBasePtr + ((y + dstY) * dstStride + dstX) * bpp;
				// If we already passed a wrap, we can use the quicker path.
				if ((srcLineStartAddr & 0x04800000) == 0x04800000)
					srcLineStartAddr &= ~0x00800000;
				if ((dstLineStartAddr & 0x04800000) == 0x04800000)
					dstLineStartAddr &= ~0x00800000;
				// These flags mean there's a wrap inside this line.
				bool srcLineWrap = !Memory::IsValidRange(srcLineStartAddr, bytesToCopy);
				bool dstLineWrap = !Memory::IsValidRange(dstLineStartAddr, bytesToCopy);

				if (!srcLineWrap && !dstLineWrap) {
					const bool lineOverlap = srcLineStartAddr < dstLineStartAddr + bytesToCopy && dstLineStartAddr < srcLineStartAddr + bytesToCopy;
					// The whole blocks the GE reads and writes.
					const u32 srcBlocksStart = srcLineStartAddr & ~15, dstBlocksStart = dstLineStartAddr & ~15;
					if (lineOverlap && Memory::IsValidRange(srcBlocksStart, ((srcLineStartAddr + bytesToCopy + 15) & ~15) - srcBlocksStart) &&
						Memory::IsValidRange(dstBlocksStart, ((dstLineStartAddr + bytesToCopy + 15) & ~15) - dstBlocksStart)) {
						BlockTransferOverlappingRow(srcLineStartAddr, dstLineStartAddr, bytesToCopy);
					} else {
						const u8 *srcp = Memory::GetPointerOrException(srcLineStartAddr);
						u8 *dstp = Memory::GetPointerWriteOrException(dstLineStartAddr);
						memmove(dstp, srcp, bytesToCopy);
					}

					// If we're tracking detail, it's useful to have the gaps illustrated properly.
					if (notifyDetail) {
						NotifyMemInfo(MemBlockFlags::READ, srcLineStartAddr, bytesToCopy, tag, tagSize);
						NotifyMemInfo(MemBlockFlags::WRITE, dstLineStartAddr, bytesToCopy, tag, tagSize);
					}
				} else {
					// We can wrap at any point, so along with overlap this gets a bit complicated.
					// We're just going to do this the slow and easy way.
					u32 srcLinePos = srcLineStartAddr;
					u32 dstLinePos = dstLineStartAddr;
					for (u32 i = 0; i < bytesToCopy; i += 64) {
						u32 chunk = i + 64 > bytesToCopy ? bytesToCopy - i : 64;
						u32 srcValid = Memory::ClampValidSizeAt(srcLinePos, chunk);
						u32 dstValid = Memory::ClampValidSizeAt(dstLinePos, chunk);

						// First chunk, for which both are valid.
						u32 bothSize = std::min(srcValid, dstValid);
						if (bothSize != 0)
							notifyingMemmove(dstLinePos, srcLinePos, bothSize);

						// Now, whichever side has more valid (or the rest, if only one side must wrap.)
						u32 exclusiveSize = std::max(srcValid, dstValid) - bothSize;
						if (exclusiveSize != 0 && srcValid >= dstValid) {
							notifyingMemmove(PSP_GetVidMemBase(), srcLineStartAddr + bothSize, exclusiveSize);
						} else if (exclusiveSize != 0 && srcValid < dstValid) {
							notifyingMemmove(dstLineStartAddr + bothSize, PSP_GetVidMemBase(), exclusiveSize);
						}

						// Finally, if both src and dst wrapped, that portion.
						u32 wrappedSize = chunk - bothSize - exclusiveSize;
						if (wrappedSize != 0 && srcValid >= dstValid) {
							notifyingMemmove(PSP_GetVidMemBase() + exclusiveSize, PSP_GetVidMemBase(), wrappedSize);
						} else if (wrappedSize != 0 && srcValid < dstValid) {
							notifyingMemmove(PSP_GetVidMemBase(), PSP_GetVidMemBase() + exclusiveSize, wrappedSize);
						}

						srcLinePos += chunk;
						dstLinePos += chunk;
						if ((srcLinePos & 0x04800000) == 0x04800000)
							srcLinePos &= ~0x00800000;
						if ((dstLinePos & 0x04800000) == 0x04800000)
							dstLinePos &= ~0x00800000;
					}
				}
			}

			if (notifyAll) {
				if (srcWraps) {
					u32 validSize = Memory::ClampValidSizeAt(src, srcSize);
					NotifyMemInfo(MemBlockFlags::READ, src, validSize, tag, tagSize);
					NotifyMemInfo(MemBlockFlags::READ, PSP_GetVidMemBase(), srcSize - validSize, tag, tagSize);
				} else {
					NotifyMemInfo(MemBlockFlags::READ, src, srcSize, tag, tagSize);
				}
				if (dstWraps) {
					u32 validSize = Memory::ClampValidSizeAt(dst, dstSize);
					NotifyMemInfo(MemBlockFlags::WRITE, dst, validSize, tag, tagSize);
					NotifyMemInfo(MemBlockFlags::WRITE, PSP_GetVidMemBase(), dstSize - validSize, tag, tagSize);
				} else {
					NotifyMemInfo(MemBlockFlags::WRITE, dst, dstSize, tag, tagSize);
				}
			}
		} else if (srcValid && dstValid) {
			u32 bytesToCopy = width * bpp;
			bool notifyDetail = MemBlockInfoDetailed(bytesToCopy);
			bool notifyAll = !notifyDetail && MemBlockInfoDetailed(srcSize, dstSize);
			if (notifyDetail || notifyAll) {
				tagSize = FormatMemWriteTagAt(tag, sizeof(tag), "GPUBlockTransfer/", src, srcSize);
			}

			for (int y = 0; y < height; y++) {
				u32 srcLineStartAddr = srcBasePtr + ((y + srcY) * srcStride + srcX) * bpp;
				u32 dstLineStartAddr = dstBasePtr + ((y + dstY) * dstStride + dstX) * bpp;

				const u8 *srcp = Memory::GetPointerOrException(srcLineStartAddr);
				u8 *dstp = Memory::GetPointerWriteOrException(dstLineStartAddr);
				memcpy(dstp, srcp, bytesToCopy);

				// If we're tracking detail, it's useful to have the gaps illustrated properly.
				if (notifyDetail) {
					NotifyMemInfo(MemBlockFlags::READ, srcLineStartAddr, bytesToCopy, tag, tagSize);
					NotifyMemInfo(MemBlockFlags::WRITE, dstLineStartAddr, bytesToCopy, tag, tagSize);
				}
			}

			if (notifyAll) {
				NotifyMemInfo(MemBlockFlags::READ, src, srcSize, tag, tagSize);
				NotifyMemInfo(MemBlockFlags::WRITE, dst, dstSize, tag, tagSize);
			}
		} else {
			// This seems to cause the GE to require a break/reset on a PSP.
			// TODO: Handle that and figure out which bytes are still copied?
			ERROR_LOG_REPORT_ONCE(invalidtransfer, Log::G3D, "Block transfer invalid: %08x/%x -> %08x/%x, %ix%ix%i (%i,%i)->(%i,%i)", srcBasePtr, srcStride, dstBasePtr, dstStride, width, height, bpp, srcX, srcY, dstX, dstY);
		}

		if (framebufferManager_) {
			// Fixes Gran Turismo's funky text issue, since it overwrites the current texture.
			textureCache_->Invalidate(dstBasePtr + (dstY * dstStride + dstX) * bpp, height * dstStride * bpp, GPU_INVALIDATE_HINT);
			framebufferManager_->NotifyBlockTransferAfter(dstBasePtr, dstStride, dstX, dstY, srcBasePtr, srcStride, srcX, srcY, width, height, bpp, skipDrawReason);
		}
	}

	// TODO: Correct timing appears to be 1.9, but erring a bit low since some of our other timing is inaccurate.
	cyclesExecuted += ((height * width * bpp) * 16) / 10;
}

void GPUCommonHWMetal::PopDLQueue() {
	if(!dlQueue.empty()) {
		dlQueue.pop_front();
		if(!dlQueue.empty()) {
			bool running = currentList->state == PSP_GE_DL_STATE_RUNNING;
			currentList = &dls[dlQueue.front()];
			if (running)
				currentList->state = PSP_GE_DL_STATE_RUNNING;
		} else {
			currentList = nullptr;
		}
	}
}

void GPUCommonHWMetal::DoExecuteCall(u32 target) {
	// Local variable for better codegen
	DisplayList *currentList = this->currentList;

	// Bone matrix optimization - many games will CALL a bone matrix (!).
	// We don't optimize during recording or debugging - so the matrix data gets recorded.
	if (useFastRunLoop_ && Memory::IsValidRange(target, 13 * 4) && (Memory::ReadUnchecked_U32(target) >> 24) == GE_CMD_BONEMATRIXDATA) {
		// Check for the end
		if ((Memory::ReadUnchecked_U32(target + 11 * 4) >> 24) == GE_CMD_BONEMATRIXDATA &&
			(Memory::ReadUnchecked_U32(target + 12 * 4) >> 24) == GE_CMD_RET &&
			(gstate.boneMatrixNumber & 0x00FFFFFF) <= 96 - 12) {
			// Yep, pretty sure this is a bone matrix call.  Double check stall first.
			if (target > currentList->stall || target + 12 * 4 < currentList->stall) {
				FastLoadBoneMatrix(target);
				return;
			}
		}
	}

	if (currentList->stackptr == ARRAY_SIZE(currentList->stack)) {
		ERROR_LOG(Log::G3D, "CALL: Stack full!");
		// TODO: UpdateState(GPUSTATE_ERROR) ?
	} else {
		auto &stackEntry = currentList->stack[currentList->stackptr++];
		stackEntry.pc = currentList->pc + 4;
		stackEntry.offsetAddr = gstate_c.offsetAddr;
		// The base address is NOT saved/restored for a regular call.
		UpdatePC(currentList->pc, target - 4);
		currentList->pc = target - 4;	// pc will be increased after we return, counteract that
	}
}

void GPUCommonHWMetal::CompleteFailedList(DisplayList &list) {
	if (list.started && list.context.IsValid()) {
		gstate.Restore(list.context);
		ReapplyGfxState();
		list.started = false;
	}
	list.state = PSP_GE_DL_STATE_COMPLETED;
	list.waitUntilTicks = startingTicks + cyclesExecuted;
	busyTicks = std::max(busyTicks, list.waitUntilTicks);
	__GeTriggerSync(GPU_SYNC_LIST, list.id, list.waitUntilTicks);
}

bool GPUCommonHWMetal::PerformMemoryCopy(u32 dest, u32 src, int size, GPUCopyFlag flags) {
	if (size == 0) {
		_dbg_assert_msg_(false, "Zero-sized PerformMemoryCopy: %08x -> %08x, size %d (flag: %d)", src, dest, size, (int)flags);
		return false;
	}

	// If dest is not a valid address, just ignore.
	if (!Memory::IsValidAddress(dest)) {
		_dbg_assert_msg_(false, "Invalid address for PerformMemorySet: %08x, size %d", dest, size);
		return false;
	}

	// Check for invalid memory range. Should we reject? For now, let's clamp it.
	if (Memory::ClampValidSizeAt(dest, size) < (u32)size) {
		ERROR_LOG_REPORT_ONCE(invalidmemset, Log::G3D, "PerformMemorySet with invalid range: %08x, size %d", dest, size);
		size = Memory::ClampValidSizeAt(dest, size);
	}

	// Track stray copies of a framebuffer in RAM. MotoGP does this.
	// Depth buffers are usually above the color range, so requested depth copies always go through.
	if ((flags & GPUCopyFlag::DEPTH_REQUESTED) || framebufferManager_->MayIntersectFramebufferColor(src) || framebufferManager_->MayIntersectFramebufferColor(dest)) {
		if (!framebufferManager_->NotifyFramebufferCopy(src, dest, size, flags, gstate_c.skipDrawReason)) {
			// We use matching values in PerformReadbackToMemory/PerformWriteColorFromMemory.
			// Since they're identical we don't need to copy.
			if (dest != src) {
				if (Memory::IsValidRange(dest, size) && Memory::IsValidRange(src, size)) {
					memcpy(Memory::GetPointerWriteUnchecked(dest), Memory::GetPointerUnchecked(src), size);
				}
				if (MemBlockInfoDetailed(size)) {
					NotifyMemInfoCopy(dest, src, size, "GPUMemcpy/");
				}
			}
		}
		InvalidateCache(dest, size, GPU_INVALIDATE_HINT);
		return true;
	}

	if (MemBlockInfoDetailed(size)) {
		NotifyMemInfoCopy(dest, src, size, "GPUMemcpy/");
	}
	InvalidateCache(dest, size, GPU_INVALIDATE_HINT);
	if (!(flags & GPUCopyFlag::DEBUG_NOTIFIED))
		recorder_.NotifyMemcpy(dest, src, size);
	return false;
}

bool GPUCommonHWMetal::PerformMemorySet(u32 dest, u8 v, int size) {
	if (size == 0) {
		_dbg_assert_msg_(false, "Zero-sized PerformMemorySet: %08x, value %02x, size %d", dest, v, size);
		return false;
	}

	// If dest is not a valid address, just ignore.
	if (!Memory::IsValidAddress(dest)) {
		_dbg_assert_msg_(false, "Invalid address for PerformMemorySet: %08x, size %d", dest, size);
		return false;
	}

	// Check for invalid memory range. Should we reject? For now, let's clamp it.
	if (Memory::ClampValidSizeAt(dest, size) < (u32)size) {
		ERROR_LOG_REPORT_ONCE(invalidmemset, Log::G3D, "PerformMemorySet with invalid range: %08x, size %d", dest, size);
		size = Memory::ClampValidSizeAt(dest, size);
	}

	// This may indicate a memset, usually to 0, of a framebuffer.
	if (framebufferManager_->MayIntersectFramebufferColor(dest)) {
		Memory::Memset(dest, v, size, "GPUMemset");
		if (!framebufferManager_->NotifyFramebufferCopy(dest, dest, size, GPUCopyFlag::MEMSET, gstate_c.skipDrawReason)) {
			InvalidateCache(dest, size, GPU_INVALIDATE_HINT);
		}
		return true;
	}

	NotifyMemInfo(MemBlockFlags::WRITE, dest, size, "GPUMemset");
	// Or perhaps a texture, let's invalidate.
	InvalidateCache(dest, size, GPU_INVALIDATE_HINT);
	recorder_.NotifyMemset(dest, v, size);
	return false;
}

bool GPUCommonHWMetal::PerformReadbackToMemory(u32 dest, int size) {
	if (Memory::IsVRAMAddress(dest)) {
		return PerformMemoryCopy(dest, dest, size, GPUCopyFlag::FORCE_DST_MATCH_MEM);
	}
	return false;
}

bool GPUCommonHWMetal::PerformWriteColorFromMemory(u32 dest, int size) {
	if (Memory::IsVRAMAddress(dest)) {
		recorder_.NotifyUpload(dest, size);
		return PerformMemoryCopy(dest, dest, size, GPUCopyFlag::FORCE_SRC_MATCH_MEM | GPUCopyFlag::DEBUG_NOTIFIED);
	}
	return false;
}

void GPUCommonHWMetal::PerformWriteFormattedFromMemory(u32 addr, int size, int frameWidth, GEBufferFormat format, int rectWidth) {
	NoteVideoRange(addr, (u32)size);
	if (Memory::IsVRAMAddress(addr)) {
		framebufferManager_->PerformWriteFormattedFromMemory(addr, size, frameWidth, format, rectWidth);
	}
	InvalidateCache(addr, size, GPU_INVALIDATE_SAFE);
}

bool GPUCommonHWMetal::PerformWriteStencilFromMemory(u32 dest, int size, WriteStencil flags) {
	if (framebufferManager_->MayIntersectFramebufferColor(dest)) {
		framebufferManager_->PerformWriteStencilFromMemory(dest, size, flags);
		return true;
	}
	return false;
}

bool GPUCommonHWMetal::PerformReadbackStencilToMemory(u32 dest, int size) {
	if (Memory::IsVRAMAddress(dest) && framebufferManager_->MayIntersectFramebufferColor(dest)) {
		return framebufferManager_->ReadbackStencilToMemory(dest, size);
	}
	return false;
}
