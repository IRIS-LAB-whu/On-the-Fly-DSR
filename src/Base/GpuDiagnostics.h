#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace GpuDiagnostics {

struct EventStats {
	std::string event;
	int imageID = -1;
	std::string role;
	std::string detail;
	std::size_t points = 0;
	std::size_t faces = 0;
	std::size_t bytes = 0;
};

struct BAStats {
	std::string baType;
	std::size_t numImages = 0;
	std::size_t numPoints = 0;
	std::size_t numVariablePoints = 0;
	std::size_t numConstantPoints = 0;
	std::size_t numResiduals = 0;
	std::size_t numObservations = 0;
	std::string linearSolverType;
	std::string denseLinearAlgebraLibraryType;
	bool useCudaSolver = false;
	int numThreads = 0;
	double initialCost = 0.0;
	double finalCost = 0.0;
	double finalCostPx = 0.0;
	double solveTimeMs = 0.0;
};

struct ProcessMemoryStats {
	double workingSetMb = -1.0;
	double peakWorkingSetMb = -1.0;
	double privateUsageMb = -1.0;
};

bool IsEnabled();
bool TimingEnabled();
void SetOutputRoot(const std::string& denseReconstructionDir);
std::string TimingDir();
std::string EscapeCsvValue(const std::string& value);
std::string Timestamp();
ProcessMemoryStats CurrentProcessMemory();
void AppendTimingRow(
	const std::string& fileName,
	const std::string& header,
	const std::string& row);
void RecordEvent(const EventStats& stats);
void RecordCurrentProcess(const std::string& role);
void RecordProcess(const std::string& role, std::uint64_t pid);
void RecordBundleAdjustment(const BAStats& stats);
void StartProcessSampler();
void StopProcessSampler();

} // namespace GpuDiagnostics
