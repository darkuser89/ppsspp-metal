// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

class GPUCommon;
class GraphicsContext;
namespace Draw { class DrawContext; }

// Keep Objective-C types out of the cross-platform GPU factory.
GPUCommon *CreateMetalGPU(GraphicsContext *context, Draw::DrawContext *draw);
