// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/StateMappingMetal.h"

namespace {

constexpr MTLBlendFactor blendFactors[] = {
	MTLBlendFactorZero, MTLBlendFactorOne,
	MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
	MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor,
	MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
	MTLBlendFactorDestinationAlpha, MTLBlendFactorOneMinusDestinationAlpha,
	MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
	MTLBlendFactorBlendAlpha, MTLBlendFactorOneMinusBlendAlpha,
	MTLBlendFactorSource1Color, MTLBlendFactorOneMinusSource1Color,
	MTLBlendFactorSource1Alpha, MTLBlendFactorOneMinusSource1Alpha,
};
static_assert(ARRAY_SIZE(blendFactors) == (size_t)BlendFactor::INVALID);

constexpr MTLBlendOperation blendOps[] = {
	MTLBlendOperationAdd, MTLBlendOperationSubtract, MTLBlendOperationReverseSubtract,
	MTLBlendOperationMin, MTLBlendOperationMax,
};
static_assert(ARRAY_SIZE(blendOps) == (size_t)BlendEq::COUNT);

constexpr MTLCompareFunction compareOps[] = {
	MTLCompareFunctionNever, MTLCompareFunctionAlways, MTLCompareFunctionEqual,
	MTLCompareFunctionNotEqual, MTLCompareFunctionLess, MTLCompareFunctionLessEqual,
	MTLCompareFunctionGreater, MTLCompareFunctionGreaterEqual,
};
constexpr MTLStencilOperation stencilOps[] = {
	MTLStencilOperationKeep, MTLStencilOperationZero, MTLStencilOperationReplace,
	MTLStencilOperationInvert, MTLStencilOperationIncrementClamp, MTLStencilOperationDecrementClamp,
	MTLStencilOperationKeep, MTLStencilOperationKeep,
};

MTLColorWriteMask ColorMask(uint8_t mask) {
	// Metal's bits are ABGR; the shared mask is RGBA from the least significant bit.
	return (MTLColorWriteMask)((mask & 1 ? MTLColorWriteMaskRed : 0) |
		(mask & 2 ? MTLColorWriteMaskGreen : 0) | (mask & 4 ? MTLColorWriteMaskBlue : 0) |
		(mask & 8 ? MTLColorWriteMaskAlpha : 0));
}

}  // namespace

bool ConvertMetalDrawState(GEPrimitiveType prim, const ComputedPipelineState &pipelineState,
	MetalDrawState *state, std::string *error) {
	*state = {};
	error->clear();
	auto &blend = state->blend;
	auto &depth = state->depthStencil;
	GenericStencilFuncState stencil;
	ConvertStencilFuncState(stencil);
	if (gstate.isModeClear()) {
		blend.writeMask = ColorMask((gstate.isClearModeColorMask() ? 7 : 0) | (gstate.isClearModeAlphaMask() ? 8 : 0));
		depth.depthWrite = gstate.isClearModeDepthMask();
		if (gstate.isClearModeAlphaMask()) {
			depth.stencilEnabled = true;
			depth.pass = depth.stencilFail = depth.depthFail = MTLStencilOperationReplace;
			depth.writeMask = stencil.writeMask;
			state->stencilRef = 255;  // Overridden by the transformed clear rectangle's alpha.
		}
		return true;
	}

	const auto &generic = pipelineState.blendState;
	blend.writeMask = ColorMask(pipelineState.maskState.channelMask);
	blend.enabled = generic.blendEnabled;
	if (blend.enabled) {
		if ((size_t)generic.srcColor >= ARRAY_SIZE(blendFactors) || (size_t)generic.dstColor >= ARRAY_SIZE(blendFactors) ||
			(size_t)generic.srcAlpha >= ARRAY_SIZE(blendFactors) || (size_t)generic.dstAlpha >= ARRAY_SIZE(blendFactors) ||
			(size_t)generic.eqColor >= ARRAY_SIZE(blendOps) || (size_t)generic.eqAlpha >= ARRAY_SIZE(blendOps)) {
			*error = "Invalid PSP blend state for Metal";
			return false;
		}
		blend.srcColor = blendFactors[(size_t)generic.srcColor];
		blend.dstColor = blendFactors[(size_t)generic.dstColor];
		blend.srcAlpha = blendFactors[(size_t)generic.srcAlpha];
		blend.dstAlpha = blendFactors[(size_t)generic.dstAlpha];
		blend.colorOp = blendOps[(size_t)generic.eqColor];
		blend.alphaOp = blendOps[(size_t)generic.eqAlpha];
		state->blendColor = generic.useBlendColor ? generic.blendColor : 0;
	}
	if (pipelineState.logicState.logicOpEnabled) {
		*error = "Metal logic operations must be resolved by the shared shader path";
		return false;
	}
	if (!IsDepthTestEffectivelyDisabled()) {
		depth.depthCompare = compareOps[gstate.getDepthTestFunction()];
		depth.depthWrite = gstate.isDepthWriteEnabled();
	}
	if (stencil.enabled) {
		depth.stencilEnabled = true;
		depth.stencilCompare = compareOps[stencil.testFunc];
		depth.pass = stencilOps[stencil.zPass];
		depth.stencilFail = stencilOps[stencil.sFail];
		depth.depthFail = stencilOps[stencil.zFail];
		depth.readMask = stencil.testMask;
		depth.writeMask = stencil.writeMask;
		state->stencilRef = stencil.testRef;
		if (SpongebobDepthInverseConditions(stencil)) {
			blend.enabled = true;
			blend.colorOp = blend.alphaOp = MTLBlendOperationAdd;
			blend.srcColor = blend.dstColor = blend.srcAlpha = blend.dstAlpha = MTLBlendFactorZero;
			blend.writeMask = MTLColorWriteMaskAlpha;
			depth.depthCompare = MTLCompareFunctionLess;
			depth.stencilCompare = MTLCompareFunctionAlways;
			depth.pass = depth.stencilFail = MTLStencilOperationZero;
			depth.depthFail = MTLStencilOperationKeep;
		}
	}
	if (prim > GE_PRIM_LINE_STRIP && prim != GE_PRIM_RECTANGLES && gstate.isCullEnabled()) {
		state->cull = gstate.getCullMode() ? MTLCullModeFront : MTLCullModeBack;
	}
	if (!gstate.isModeThrough() && (gstate.getDepthRangeMin() == 0 || gstate.getDepthRangeMax() == 65535) &&
		gstate.isDepthClipEnabled() && gstate_c.Use(GPU_USE_DEPTH_CLAMP)) {
		state->depthClip = MTLDepthClipModeClamp;
	}
	return true;
}
