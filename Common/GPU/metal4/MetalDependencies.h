// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once

#include <cstdint>
#include <map>

namespace Metal {

struct ResourceAccess {
	uint64_t reads = 0;
	uint64_t writes = 0;
};

// Keys identify the backing allocation, including all views, mips and layers.
using ResourceAccesses = std::map<uintptr_t, ResourceAccess>;

struct PassDependencies {
	uint64_t afterStages = 0;
	uint64_t beforeStages = 0;
	uint64_t raw = 0;
	uint64_t war = 0;
	uint64_t waw = 0;
	void Include(const PassDependencies &other) {
		afterStages |= other.afterStages;
		beforeStages |= other.beforeStages;
		raw += other.raw;
		war += other.war;
		waw += other.waw;
	}
};

inline PassDependencies FindDependencies(const ResourceAccess &prior, const ResourceAccess &access) {
	PassDependencies result;
	if (prior.writes && access.reads) {
		++result.raw;
		result.afterStages |= prior.writes;
		result.beforeStages |= access.reads;
	}
	if (prior.reads && access.writes) {
		++result.war;
		result.afterStages |= prior.reads;
		result.beforeStages |= access.writes;
	}
	if (prior.writes && access.writes) {
		++result.waw;
		result.afterStages |= prior.writes;
		result.beforeStages |= access.writes;
	}
	return result;
}

inline PassDependencies FindDependencies(const ResourceAccesses &previous, const ResourceAccesses &current) {
	PassDependencies result;
	for (const auto &[resource, access] : current) {
		auto found = previous.find(resource);
		if (found == previous.end()) {
			continue;
		}
		result.Include(FindDependencies(found->second, access));
	}
	return result;
}

inline void MergeAccesses(ResourceAccesses *previous, const ResourceAccesses &current) {
	for (const auto &[resource, access] : current) {
		auto &prior = (*previous)[resource];
		// Keep all potentially concurrent readers/writers. A stage barrier can
		// cover more than this resource, but is not a resource completion event.
		prior.reads |= access.reads;
		prior.writes |= access.writes;
	}
}

}  // namespace Metal
