#include "GpuDiagnostics.h"

#include "config.h"
#include "misc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

namespace GpuDiagnostics {
namespace {

std::mutex g_mutex;
std::string g_outputRoot;
std::atomic_bool g_samplerRunning{ false };
std::thread g_samplerThread;
std::unordered_map<std::uint64_t, std::string> g_processRoles;

std::string EscapeCsv(std::string value)
{
	const bool quote = value.find_first_of(",\"\r\n") != std::string::npos;
	if (!quote) {
		return value;
	}
	std::string escaped = "\"";
	for (const char c : value) {
		if (c == '"') {
			escaped += "\"\"";
		}
		else {
			escaped += c;
		}
	}
	escaped += "\"";
	return escaped;
}

std::string NowString()
{
	const auto now = std::chrono::system_clock::now();
	const auto time = std::chrono::system_clock::to_time_t(now);
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		now.time_since_epoch()) % 1000;
	std::tm tm{};
#ifdef _WIN32
	localtime_s(&tm, &time);
#else
	localtime_r(&time, &tm);
#endif
	std::ostringstream stream;
	stream << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
		<< "." << std::setfill('0') << std::setw(3) << ms.count();
	return stream.str();
}

std::uint64_t CurrentPid()
{
#ifdef _WIN32
	return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
	return 0;
#endif
}

std::string ThreadIdString()
{
	std::ostringstream stream;
	stream << std::this_thread::get_id();
	return stream.str();
}

std::string RunCommand(const std::string& command)
{
	std::string output;
#ifdef _WIN32
	SECURITY_ATTRIBUTES securityAttributes{};
	securityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
	securityAttributes.bInheritHandle = TRUE;
	securityAttributes.lpSecurityDescriptor = nullptr;

	HANDLE readPipe = nullptr;
	HANDLE writePipe = nullptr;
	if (!CreatePipe(&readPipe, &writePipe, &securityAttributes, 0)) {
		return output;
	}
	if (!SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0)) {
		CloseHandle(readPipe);
		CloseHandle(writePipe);
		return output;
	}

	STARTUPINFOA startupInfo{};
	startupInfo.cb = sizeof(STARTUPINFOA);
	startupInfo.dwFlags = STARTF_USESTDHANDLES;
	startupInfo.hStdOutput = writePipe;
	startupInfo.hStdError = writePipe;
	startupInfo.hStdInput = nullptr;

	PROCESS_INFORMATION processInfo{};
	std::string commandLine = "cmd.exe /C " + command;
	if (!CreateProcessA(
		nullptr,
		commandLine.data(),
		nullptr,
		nullptr,
		TRUE,
		CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&startupInfo,
		&processInfo)) {
		CloseHandle(readPipe);
		CloseHandle(writePipe);
		return output;
	}

	CloseHandle(writePipe);
	char buffer[512];
	DWORD bytesRead = 0;
	while (ReadFile(readPipe, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) &&
		bytesRead > 0) {
		buffer[bytesRead] = '\0';
		output += buffer;
	}
	WaitForSingleObject(processInfo.hProcess, INFINITE);
	CloseHandle(processInfo.hThread);
	CloseHandle(processInfo.hProcess);
	CloseHandle(readPipe);
#else
	FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		return output;
	}
	char buffer[512];
	while (fgets(buffer, sizeof(buffer), pipe)) {
		output += buffer;
	}
	pclose(pipe);
#endif
	return output;
}

struct GpuMemoryInfo {
	double totalMb = -1.0;
	double usedMb = -1.0;
	double freeMb = -1.0;
};

GpuMemoryInfo QueryGpuMemory()
{
	GpuMemoryInfo info;
	if (!IsEnabled() || !Config::getInstance().gpuDiagnostics.UseNvidiaSmi) {
		return info;
	}
	const std::string output = RunCommand(
		"nvidia-smi --query-gpu=memory.total,memory.used,memory.free "
		"--format=csv,noheader,nounits");
	std::istringstream lines(output);
	std::string line;
	if (!std::getline(lines, line)) {
		return info;
	}
	std::replace(line.begin(), line.end(), ',', ' ');
	std::istringstream values(line);
	values >> info.totalMb >> info.usedMb >> info.freeMb;
	return info;
}

void AppendLine(
	const std::string& fileName,
	const std::string& header,
	const std::string& line)
{
	if (g_outputRoot.empty()) {
		return;
	}
	const std::string dir = TimingDir();
	CreateDirIfNotExists(dir, true);
	const std::string path = dir + "/" + fileName;
	const bool writeHeader = !static_cast<bool>(std::ifstream(path));
	std::ofstream file(path, std::ios::app);
	if (!file) {
		return;
	}
	if (writeHeader) {
		file << header << "\n";
	}
	file << line << "\n";
}

void SampleProcessMemory()
{
	if (!Config::getInstance().gpuDiagnostics.UseNvidiaSmi) {
		return;
	}
	std::unordered_map<std::uint64_t, std::string> roles;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		roles = g_processRoles;
	}
	if (roles.empty()) {
		return;
	}

	const std::string output = RunCommand(
		"nvidia-smi --query-compute-apps=pid,process_name,used_memory "
		"--format=csv,noheader,nounits");
	std::istringstream lines(output);
	std::string line;
	while (std::getline(lines, line)) {
		if (line.empty()) {
			continue;
		}
		std::vector<std::string> parts;
		std::stringstream ss(line);
		std::string part;
		while (std::getline(ss, part, ',')) {
			const auto first = part.find_first_not_of(" \t\r\n");
			const auto last = part.find_last_not_of(" \t\r\n");
			parts.push_back(first == std::string::npos
				? ""
				: part.substr(first, last - first + 1));
		}
		if (parts.size() < 3) {
			continue;
		}
		const std::uint64_t pid = std::strtoull(parts[0].c_str(), nullptr, 10);
		const auto roleIt = roles.find(pid);
		if (roleIt == roles.end()) {
			continue;
		}
		const std::string header =
			"timestamp,pid,role,process_name,used_memory_mb";
		const std::string row =
			EscapeCsv(NowString()) + "," +
			std::to_string(pid) + "," +
			EscapeCsv(roleIt->second) + "," +
			EscapeCsv(parts[1]) + "," +
			parts[2];
		std::lock_guard<std::mutex> lock(g_mutex);
		AppendLine("gpu_memory_process.csv", header, row);
	}
}

} // namespace

bool IsEnabled()
{
	const auto& config = Config::getInstance();
	return config.pipeline.OutputTimingFiles && config.gpuDiagnostics.Enable;
}

bool TimingEnabled()
{
	return Config::getInstance().pipeline.OutputTimingFiles;
}

void SetOutputRoot(const std::string& denseReconstructionDir)
{
	if (!TimingEnabled() || denseReconstructionDir.empty()) {
		return;
	}
	std::lock_guard<std::mutex> lock(g_mutex);
	g_outputRoot = StringReplace(EnsureTrailingSlash(denseReconstructionDir), "\\", "/");
	CreateDirIfNotExists(TimingDir(), true);
}

std::string TimingDir()
{
	return StringReplace(EnsureTrailingSlash(g_outputRoot), "\\", "/") + "Timing";
}

std::string EscapeCsvValue(const std::string& value)
{
	return EscapeCsv(value);
}

std::string Timestamp()
{
	return NowString();
}

ProcessMemoryStats CurrentProcessMemory()
{
	ProcessMemoryStats stats;
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS_EX counters{};
	counters.cb = sizeof(counters);
	if (GetProcessMemoryInfo(
		GetCurrentProcess(),
		reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
		sizeof(counters))) {
		stats.workingSetMb =
			static_cast<double>(counters.WorkingSetSize) / (1024.0 * 1024.0);
		stats.peakWorkingSetMb =
			static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
		stats.privateUsageMb =
			static_cast<double>(counters.PrivateUsage) / (1024.0 * 1024.0);
	}
#endif
	return stats;
}

void AppendTimingRow(
	const std::string& fileName,
	const std::string& header,
	const std::string& row)
{
	if (!TimingEnabled()) {
		return;
	}
	std::lock_guard<std::mutex> lock(g_mutex);
	AppendLine(fileName, header, row);
}

void RecordEvent(const EventStats& stats)
{
	if (!IsEnabled() || !Config::getInstance().gpuDiagnostics.LogEventSnapshots) {
		return;
	}
	const GpuMemoryInfo gpu = QueryGpuMemory();
	const std::string header =
		"timestamp,pid,thread,event,image_id,role,detail,points,faces,bytes,"
		"gpu_total_mb,gpu_used_mb,gpu_free_mb";
	const std::string row =
		EscapeCsv(NowString()) + "," +
		std::to_string(CurrentPid()) + "," +
		EscapeCsv(ThreadIdString()) + "," +
		EscapeCsv(stats.event) + "," +
		std::to_string(stats.imageID) + "," +
		EscapeCsv(stats.role) + "," +
		EscapeCsv(stats.detail) + "," +
		std::to_string(stats.points) + "," +
		std::to_string(stats.faces) + "," +
		std::to_string(stats.bytes) + "," +
		std::to_string(gpu.totalMb) + "," +
		std::to_string(gpu.usedMb) + "," +
		std::to_string(gpu.freeMb);
	std::lock_guard<std::mutex> lock(g_mutex);
	AppendLine("gpu_memory_main.csv", header, row);
}

void RecordCurrentProcess(const std::string& role)
{
	RecordProcess(role, CurrentPid());
}

void RecordProcess(const std::string& role, const std::uint64_t pid)
{
	if (!TimingEnabled() || !Config::getInstance().gpuDiagnostics.Enable || pid == 0) {
		return;
	}
	const std::string header = "timestamp,pid,role";
	const std::string row =
		EscapeCsv(NowString()) + "," +
		std::to_string(pid) + "," +
		EscapeCsv(role);
	std::lock_guard<std::mutex> lock(g_mutex);
	g_processRoles[pid] = role;
	AppendLine("gpu_process_map.csv", header, row);
}

void RecordBundleAdjustment(const BAStats& stats)
{
	if (!TimingEnabled() || !Config::getInstance().gpuDiagnostics.LogBundleAdjustmentStats) {
		return;
	}
	const GpuMemoryInfo gpu = QueryGpuMemory();
	const std::string header =
		"timestamp,ba_type,num_images,num_points,num_variable_points,"
		"num_constant_points,num_observations,num_residuals,linear_solver,"
		"dense_linear_algebra,use_cuda_solver,num_threads,initial_cost,"
		"final_cost,final_cost_px,solve_time_ms,gpu_total_mb,gpu_used_mb,"
		"gpu_free_mb";
	const std::string row =
		EscapeCsv(NowString()) + "," +
		EscapeCsv(stats.baType) + "," +
		std::to_string(stats.numImages) + "," +
		std::to_string(stats.numPoints) + "," +
		std::to_string(stats.numVariablePoints) + "," +
		std::to_string(stats.numConstantPoints) + "," +
		std::to_string(stats.numObservations) + "," +
		std::to_string(stats.numResiduals) + "," +
		EscapeCsv(stats.linearSolverType) + "," +
		EscapeCsv(stats.denseLinearAlgebraLibraryType) + "," +
		(stats.useCudaSolver ? "true" : "false") + "," +
		std::to_string(stats.numThreads) + "," +
		std::to_string(stats.initialCost) + "," +
		std::to_string(stats.finalCost) + "," +
		std::to_string(stats.finalCostPx) + "," +
		std::to_string(stats.solveTimeMs) + "," +
		std::to_string(gpu.totalMb) + "," +
		std::to_string(gpu.usedMb) + "," +
		std::to_string(gpu.freeMb);
	std::lock_guard<std::mutex> lock(g_mutex);
	AppendLine("ba_timing_stats.csv", header, row);
}

void StartProcessSampler()
{
	if (!IsEnabled() || !Config::getInstance().gpuDiagnostics.UseNvidiaSmi) {
		return;
	}
	bool expected = false;
	if (!g_samplerRunning.compare_exchange_strong(expected, true)) {
		return;
	}
	g_samplerThread = std::thread([]() {
		while (g_samplerRunning.load()) {
			SampleProcessMemory();
			const int intervalMs =
				std::max(500, Config::getInstance().gpuDiagnostics.PollIntervalMs);
			std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
		}
	});
}

void StopProcessSampler()
{
	if (!g_samplerRunning.exchange(false)) {
		return;
	}
	if (g_samplerThread.joinable()) {
		g_samplerThread.join();
	}
}

} // namespace GpuDiagnostics
