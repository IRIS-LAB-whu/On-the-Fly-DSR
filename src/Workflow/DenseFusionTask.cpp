#include "DenseFusionTask.h"
#include "DepthAlignmentTask.h"

#include "../Base/config.h"
#include "../Base/GpuDiagnostics.h"
#include "../Base/misc.h"
#include "../Base/timer.h"
#include "../Feature/Bitmap.h"
#include "../Scene/Database.h"
#include "../Scene/ReconstructionManager.h"
#include "WorkData.h"

#include <glog/logging.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/Image.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/VoxelBlockGrid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool UseImageColor();
Eigen::Vector3d DenseMeshWhiteColor();

struct NpyFloatImage {
	int width = 0;
	int height = 0;
	std::vector<float> data;
};

struct TsdfRuntimeStats {
	int validDepthPixels = 0;
	double integrateMs = 0.0;
	double extractMs = 0.0;
	double totalMs = 0.0;
	size_t displayPoints = 0;
	size_t meshVertices = 0;
	size_t meshTriangles = 0;
	bool weighted = false;
	std::string globalP75Metric = "none";
	double meanEffectiveWeight = 1.0;
	double hardRejectRatio = 0.0;
	double softSuppressionRatio = 0.0;
	double upgradeRatio = 0.0;
	double weightedIntegrateMs = 0.0;
	double globalResidualReference = 0.0;
	double globalResidualReferenceQuantile = 0.0;
	size_t globalResidualSampleCount = 0;
	int minUpgradeSamples = 0;
	double upgradeEligibleRatio = 0.0;
	double upgradeBlockedLowSamplesRatio = 0.0;
	double historyUpdateEligibleRatio = 0.0;
	double historyUpdateExcludedRejectRatio = 0.0;
	double historyUpdateExcludedUnsupportedRatio = 0.0;
	double upgradeBoostRatio = 0.0;
	double meanUpgradeMultiplier = 0.0;
	double maxEffectiveWeight = 0.0;
	double upgradeCapDivisor = 0.0;
	double upgradeWeightCap = 0.0;
	double extractionWeightThreshold = 3.0;
};

std::string ReadExact(std::ifstream& stream, const size_t size)
{
	std::string buffer(size, '\0');
	stream.read(buffer.data(), static_cast<std::streamsize>(size));
	if (stream.gcount() != static_cast<std::streamsize>(size)) {
		throw std::runtime_error("Unexpected end of npy file.");
	}
	return buffer;
}

uint16_t ReadLittleEndianU16(std::ifstream& stream)
{
	const std::string bytes = ReadExact(stream, 2);
	return static_cast<uint16_t>(
		static_cast<unsigned char>(bytes[0]) |
		(static_cast<unsigned char>(bytes[1]) << 8));
}

uint32_t ReadLittleEndianU32(std::ifstream& stream)
{
	const std::string bytes = ReadExact(stream, 4);
	return static_cast<uint32_t>(
		static_cast<unsigned char>(bytes[0]) |
		(static_cast<unsigned char>(bytes[1]) << 8) |
		(static_cast<unsigned char>(bytes[2]) << 16) |
		(static_cast<unsigned char>(bytes[3]) << 24));
}

std::vector<int> ParseShape(const std::string& header)
{
	const size_t begin = header.find('(');
	const size_t end = header.find(')', begin);
	if (begin == std::string::npos || end == std::string::npos || end <= begin) {
		throw std::runtime_error("Invalid npy shape header.");
	}

	std::vector<int> shape;
	std::stringstream ss(header.substr(begin + 1, end - begin - 1));
	while (ss.good()) {
		std::string item;
		std::getline(ss, item, ',');
		item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
		if (!item.empty()) {
			shape.push_back(std::stoi(item));
		}
	}
	return shape;
}

std::string ParentPath(const std::string& path)
{
	const size_t pos = path.find_last_of("/\\");
	if (pos == std::string::npos) {
		return "";
	}
	return path.substr(0, pos);
}

std::string FileStem(const std::string& path)
{
	const size_t slash = path.find_last_of("/\\");
	const size_t begin = slash == std::string::npos ? 0 : slash + 1;
	const size_t dot = path.find_last_of('.');
	const size_t end = dot == std::string::npos || dot < begin ? path.size() : dot;
	return path.substr(begin, end - begin);
}

std::string NormalizePathSeparators(std::string path)
{
	std::replace(path.begin(), path.end(), '\\', '/');
	return path;
}

int ToPlyColor(const double value)
{
	return static_cast<int>(std::clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

bool WritePointCloudPly(
	const open3d::geometry::PointCloud& pointCloud,
	const std::string& path)
{
	if (pointCloud.IsEmpty()) {
		return false;
	}

	std::ofstream file(path);
	if (!file) {
		throw std::runtime_error("Cannot write dense point cloud PLY: " + path);
	}

	file << "ply\n"
		<< "format ascii 1.0\n"
		<< "element vertex " << pointCloud.points_.size() << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property uchar red\n"
		<< "property uchar green\n"
		<< "property uchar blue\n"
		<< "end_header\n";

	for (size_t i = 0; i < pointCloud.points_.size(); ++i) {
		const Eigen::Vector3d color = i < pointCloud.colors_.size()
			? pointCloud.colors_[i]
			: DenseMeshWhiteColor();
		const Eigen::Vector3d& point = pointCloud.points_[i];
		file << point.x() << ' ' << point.y() << ' ' << point.z() << ' '
			<< ToPlyColor(color.x()) << ' '
			<< ToPlyColor(color.y()) << ' '
			<< ToPlyColor(color.z()) << '\n';
	}
	return true;
}

bool WriteTriangleMeshPly(
	open3d::geometry::TriangleMesh mesh,
	const std::string& path)
{
	if (mesh.IsEmpty()) {
		return false;
	}

	mesh.ComputeVertexNormals();
	std::ofstream file(path);
	if (!file) {
		throw std::runtime_error("Cannot write dense mesh PLY: " + path);
	}

	file << "ply\n"
		<< "format ascii 1.0\n"
		<< "element vertex " << mesh.vertices_.size() << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property uchar red\n"
		<< "property uchar green\n"
		<< "property uchar blue\n"
		<< "element face " << mesh.triangles_.size() << "\n"
		<< "property list uchar int vertex_indices\n"
		<< "end_header\n";

	const bool useImageColor = UseImageColor();
	for (size_t i = 0; i < mesh.vertices_.size(); ++i) {
		const Eigen::Vector3d color = useImageColor && i < mesh.vertex_colors_.size()
			? mesh.vertex_colors_[i]
			: DenseMeshWhiteColor();
		const Eigen::Vector3d& point = mesh.vertices_[i];
		file << point.x() << ' ' << point.y() << ' ' << point.z() << ' '
			<< ToPlyColor(color.x()) << ' '
			<< ToPlyColor(color.y()) << ' '
			<< ToPlyColor(color.z()) << '\n';
	}
	for (const Eigen::Vector3i& tri : mesh.triangles_) {
		file << "3 " << tri.x() << ' ' << tri.y() << ' ' << tri.z() << '\n';
	}
	return true;
}

std::string DepthSidecarPath(
	const std::string& depthPath,
	const std::string& folder,
	const std::string& suffix)
{
	const std::string npDir = ParentPath(depthPath);
	const std::string rootDir = ParentPath(npDir);
	std::string stem = FileStem(depthPath);
	const std::string depthSuffix = "_depth";
	if (stem.size() > depthSuffix.size() &&
		stem.compare(stem.size() - depthSuffix.size(), depthSuffix.size(), depthSuffix) == 0) {
		stem.resize(stem.size() - depthSuffix.size());
	}
	const char sep = rootDir.find('\\') != std::string::npos ? '\\' : '/';
	return rootDir + sep + folder + sep + stem + suffix;
}

std::vector<double> ReadTextMatrixValues(
	const std::string& path,
	const size_t expectedValues)
{
	std::ifstream stream(path);
	if (!stream) {
		throw std::runtime_error("Cannot open dense fusion sidecar file: " + path);
	}
	std::vector<double> values;
	values.reserve(expectedValues);
	double value = 0.0;
	while (stream >> value) {
		values.push_back(value);
	}
	if (values.size() != expectedValues) {
		throw std::runtime_error(
			"Invalid dense fusion sidecar matrix size: " + path);
	}
	return values;
}

open3d::core::Tensor ReadTensorMatrix(
	const std::string& path,
	const int rows,
	const int cols)
{
	return open3d::core::Tensor(
		ReadTextMatrixValues(path, static_cast<size_t>(rows * cols)),
		{ rows, cols },
		open3d::core::Dtype::Float64,
		open3d::core::Device("CPU:0"));
}

open3d::core::Tensor MakeTensorMatrix(
	const std::vector<double>& values,
	const int rows,
	const int cols)
{
	return open3d::core::Tensor(
		values,
		{ rows, cols },
		open3d::core::Dtype::Float64,
		open3d::core::Device("CPU:0"));
}

NpyFloatImage ReadNpyFloatImage(const std::string& path)
{
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Cannot open depth npy: " + path);
	}

	const std::string magic = ReadExact(stream, 6);
	if (magic != "\x93NUMPY") {
		throw std::runtime_error("Invalid npy magic: " + path);
	}

	const uint8_t major = static_cast<uint8_t>(ReadExact(stream, 1)[0]);
	ReadExact(stream, 1);

	uint32_t headerLength = 0;
	if (major == 1) {
		headerLength = ReadLittleEndianU16(stream);
	}
	else if (major == 2 || major == 3) {
		headerLength = ReadLittleEndianU32(stream);
	}
	else {
		throw std::runtime_error("Unsupported npy version.");
	}

	const std::string header = ReadExact(stream, headerLength);
	if (header.find("'descr': '<f4'") == std::string::npos &&
		header.find("\"descr\": \"<f4\"") == std::string::npos) {
		throw std::runtime_error("Only little-endian float32 npy depth is supported.");
	}
	if (header.find("True") != std::string::npos) {
		throw std::runtime_error("Fortran-order npy depth is not supported.");
	}

	const std::vector<int> shape = ParseShape(header);
	if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
		throw std::runtime_error("Depth npy must have shape HxW.");
	}

	NpyFloatImage image;
	image.height = shape[0];
	image.width = shape[1];
	image.data.resize(static_cast<size_t>(image.width) * image.height);
	stream.read(
		reinterpret_cast<char*>(image.data.data()),
		static_cast<std::streamsize>(image.data.size() * sizeof(float)));
	if (stream.gcount() != static_cast<std::streamsize>(image.data.size() * sizeof(float))) {
		throw std::runtime_error("Depth npy payload is incomplete.");
	}
	return image;
}

int CountValidDepthPixels(const NpyFloatImage& depth, const float depthMax)
{
	return static_cast<int>(std::count_if(
		depth.data.begin(),
		depth.data.end(),
		[depthMax](const float value) {
			return std::isfinite(value) && value > 0.0f && value < depthMax;
		}));
}

open3d::geometry::Image MakeDepthImage(const NpyFloatImage& npy)
{
	open3d::geometry::Image depth;
	depth.Prepare(npy.width, npy.height, 1, 4);
	std::memcpy(depth.data_.data(), npy.data.data(), npy.data.size() * sizeof(float));
	return depth;
}

open3d::t::geometry::Image MakeTensorDepthImage(
	const NpyFloatImage& npy,
	const open3d::core::Device& device)
{
	open3d::core::Tensor depth(
		{ npy.height, npy.width },
		open3d::core::Dtype::Float32,
		open3d::core::Device("CPU:0"));
	std::memcpy(depth.GetDataPtr(), npy.data.data(), npy.data.size() * sizeof(float));
	return open3d::t::geometry::Image(depth.To(device));
}

open3d::t::geometry::Image MakeTensorFloatMapImage(
	const std::vector<float>& values,
	const int width,
	const int height,
	const char* mapName,
	const open3d::core::Device& device)
{
	if (width <= 0 || height <= 0 ||
		values.size() != static_cast<size_t>(width) * height) {
		throw std::runtime_error(
			std::string("Alignment ") + mapName + " map has invalid dimensions.");
	}
	open3d::core::Tensor tensor(
		{ height, width, 1 },
		open3d::core::Dtype::Float32,
		open3d::core::Device("CPU:0"));
	std::memcpy(
		tensor.GetDataPtr(),
		values.data(),
		values.size() * sizeof(float));
	return open3d::t::geometry::Image(tensor.To(device));
}

std::string ToLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

bool UseImageColor()
{
	const std::string mode = ToLower(Config::getInstance().denseFusion.ColorMode);
	return mode == "image" || mode == "texture" || mode == "rgb";
}

Eigen::Vector3d DenseMeshWhiteColor()
{
	return Eigen::Vector3d(0.82, 0.82, 0.78);
}

bool GenerateDensePoints()
{
	const std::string mode = ToLower(Config::getInstance().denseFusion.OutputMode);
	return mode == "pointcloud" || mode == "point" || mode == "points" || mode == "both";
}

bool GenerateDenseMesh()
{
	const std::string mode = ToLower(Config::getInstance().denseFusion.OutputMode);
	return mode == "mesh" || mode == "both";
}

open3d::geometry::Image MakeImageColorImage(
	const std::string& imagePath,
	const int width,
	const int height)
{
	Bitmap bitmap;
	if (!bitmap.Read(imagePath, true)) {
		throw std::runtime_error("Cannot read source image for TSDF color: " + imagePath);
	}
	if (bitmap.Width() != width || bitmap.Height() != height) {
		bitmap.Rescale(width, height);
	}

	open3d::geometry::Image color;
	color.Prepare(width, height, 3, 1);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			BitmapColor<uint8_t> pixel;
			if (!bitmap.GetPixel(x, y, &pixel)) {
				throw std::runtime_error("Cannot sample source image color for TSDF.");
			}
			const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
			color.data_[offset] = pixel.r;
			color.data_[offset + 1] = pixel.g;
			color.data_[offset + 2] = pixel.b;
		}
	}
	return color;
}

open3d::t::geometry::Image MakeTensorColorImage(
	const std::string& imagePath,
	const int width,
	const int height,
	const open3d::core::Device& device)
{
	if (width <= 0 || height <= 0) {
		throw std::runtime_error("Invalid depth image dimensions for TSDF color.");
	}
	Bitmap bitmap;
	if (!bitmap.Read(imagePath, true)) {
		throw std::runtime_error("Cannot read source image for TSDF color: " + imagePath);
	}
	if (bitmap.Width() != width || bitmap.Height() != height) {
		bitmap.Rescale(width, height);
	}

	std::vector<float> colorData(static_cast<size_t>(width) * height * 3);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			BitmapColor<uint8_t> pixel;
			if (!bitmap.GetPixel(x, y, &pixel)) {
				throw std::runtime_error("Cannot sample source image color for TSDF.");
			}
			const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
			colorData[offset] = static_cast<float>(pixel.r) / 255.0f;
			colorData[offset + 1] = static_cast<float>(pixel.g) / 255.0f;
			colorData[offset + 2] = static_cast<float>(pixel.b) / 255.0f;
		}
	}

	open3d::core::Tensor color(
		{ height, width, 3 },
		open3d::core::Dtype::Float32,
		open3d::core::Device("CPU:0"));
	std::memcpy(color.GetDataPtr(), colorData.data(), colorData.size() * sizeof(float));
	return open3d::t::geometry::Image(color.To(device));
}

open3d::camera::PinholeCameraIntrinsic MakeIntrinsic(
	const Camera& camera,
	const int depthWidth,
	const int depthHeight)
{
	const double scale = std::min(
		static_cast<double>(depthWidth) / static_cast<double>(camera.Width()),
		static_cast<double>(depthHeight) / static_cast<double>(camera.Height()));
	const Eigen::Matrix3d calibration = camera.CalibrationMatrix();

	return open3d::camera::PinholeCameraIntrinsic(
		depthWidth,
		depthHeight,
		calibration(0, 0) * scale,
		calibration(1, 1) * scale,
		calibration(0, 2) * scale,
		calibration(1, 2) * scale);
}

open3d::core::Tensor MakeTensorIntrinsic(
	const Camera& camera,
	const int depthWidth,
	const int depthHeight)
{
	const open3d::camera::PinholeCameraIntrinsic intrinsic =
		MakeIntrinsic(camera, depthWidth, depthHeight);
	const Eigen::Matrix3d k = intrinsic.intrinsic_matrix_;
	std::vector<double> values = {
		k(0, 0), k(0, 1), k(0, 2),
		k(1, 0), k(1, 1), k(1, 2),
		k(2, 0), k(2, 1), k(2, 2)
	};
	return open3d::core::Tensor(
		values,
		{ 3, 3 },
		open3d::core::Dtype::Float64,
		open3d::core::Device("CPU:0"));
}

Eigen::Matrix4d MakeExtrinsic(const Image& image)
{
	Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
	extrinsic.block<3, 4>(0, 0) = image.CamFromWorld().ToMatrix();
	return extrinsic;
}

open3d::core::Tensor MakeTensorExtrinsic(const Image& image)
{
	const Eigen::Matrix4d extrinsic = MakeExtrinsic(image);
	std::vector<double> values = {
		extrinsic(0, 0), extrinsic(0, 1), extrinsic(0, 2), extrinsic(0, 3),
		extrinsic(1, 0), extrinsic(1, 1), extrinsic(1, 2), extrinsic(1, 3),
		extrinsic(2, 0), extrinsic(2, 1), extrinsic(2, 2), extrinsic(2, 3),
		extrinsic(3, 0), extrinsic(3, 1), extrinsic(3, 2), extrinsic(3, 3)
	};
	return open3d::core::Tensor(
		values,
		{ 4, 4 },
		open3d::core::Dtype::Float64,
		open3d::core::Device("CPU:0"));
}

PointPainterData MakePointData(
	const Eigen::Vector3d& p,
	const Eigen::Vector3d& c,
	const float alpha)
{
	return {
		static_cast<float>(p.x()),
		static_cast<float>(p.y()),
		static_cast<float>(p.z()),
		static_cast<float>(std::clamp(c.x(), 0.0, 1.0)),
		static_cast<float>(std::clamp(c.y(), 0.0, 1.0)),
		static_cast<float>(std::clamp(c.z(), 0.0, 1.0)),
		alpha
	};
}

LinePainterData MakeLineData(const PointPainterData& a, const PointPainterData& b)
{
	LinePainterData line{};
	std::copy(a.begin(), a.end(), line.begin());
	std::copy(b.begin(), b.end(), line.begin() + 7);
	return line;
}

TrianglePainterData MakeTriangleData(
	const PointPainterData& a,
	const PointPainterData& b,
	const PointPainterData& c)
{
	TrianglePainterData tri{};
	std::copy(a.begin(), a.end(), tri.begin());
	std::copy(b.begin(), b.end(), tri.begin() + 7);
	std::copy(c.begin(), c.end(), tri.begin() + 14);
	return tri;
}

class DenseTsdfFusion {
public:
	DenseTsdfFusion()
	{
		ResetVolume();
	}

	bool Integrate(
		const image_t imageID,
		const std::string& depthPath,
		const std::string& imagePath,
		RenderData& renderData,
		int& integratedFrames,
		int& vertices,
		int& triangles,
		bool& renderUpdated,
		const std::shared_ptr<const AlignmentQualityData>& qualityData,
		TsdfRuntimeStats* runtimeStats = nullptr)
	{
		Timer totalTimer;
		totalTimer.Start();
		std::lock_guard<std::mutex> lock(mutex_);
		renderUpdated = false;
		if (integratedImages_.count(imageID) > 0) {
			throw std::runtime_error("Image is already integrated into dense TSDF.");
		}

		const NpyFloatImage npyDepth = ReadNpyFloatImage(depthPath);
		const auto& denseFusion = Config::getInstance().denseFusion;
		const float depthScale = 1.0f;
		const float depthMax = static_cast<float>(denseFusion.DepthMax);

		const int validDepthPixels = CountValidDepthPixels(npyDepth, depthMax);
		if (runtimeStats) {
			runtimeStats->validDepthPixels = validDepthPixels;
		}
		if (validDepthPixels < denseFusion.MinValidDepthPixels) {
			LOG(WARNING) << "[DenseFusion] Skipping image " << imageID
				<< ": too few valid depth pixels (" << validDepthPixels
				<< " < " << denseFusion.MinValidDepthPixels << ").";
			integratedImages_.insert(imageID);
			integratedFrames = integratedFrameCount_;
			vertices = static_cast<int>(lastVertexCount_);
			triangles = static_cast<int>(lastTriangleCount_);
			return false;
		}

		const std::string intrinsicPath = DepthSidecarPath(depthPath, "intrinsic", ".txt");
		const std::string posePath = DepthSidecarPath(depthPath, "pose", ".txt");
		const std::vector<double> intrinsicValues = ReadTextMatrixValues(intrinsicPath, 9);
		open3d::t::geometry::Image depth = MakeTensorDepthImage(npyDepth, device_);
		open3d::t::geometry::Image color =
			MakeTensorColorImage(imagePath, npyDepth.width, npyDepth.height, device_);
		const open3d::core::Tensor intrinsic =
			MakeTensorMatrix(intrinsicValues, 3, 3);
		const open3d::core::Tensor extrinsic =
			ReadTensorMatrix(posePath, 4, 4);

		const float truncVoxelMultiplier =
			static_cast<float>(denseFusion.TruncVoxelMultiplier);
		Timer integrateTimer;
		integrateTimer.Start();
		open3d::core::Tensor blocks = volume_->GetUniqueBlockCoordinates(
			depth,
			intrinsic,
			extrinsic,
			depthScale,
			depthMax,
			truncVoxelMultiplier);
		const auto& weighted = Config::getInstance().weightedTSDF;
		if (weighted.Enable) {
			constexpr double globalReferenceQuantile = 0.75;
			const AlignmentQualityData* selectedQuality = qualityData.get();
			if (!selectedQuality) {
				LOG(WARNING) << "[WeightedTSDF] Image " << imageID
					<< " has no AlignGrid quality data; skipping integration.";
				integratedImages_.insert(imageID);
				return false;
			}
			for (const float value : selectedQuality->windowFusionMetrics) {
				if (std::isfinite(value) && value >= 0.0f) {
					AddGlobalResidualMetric(value);
				}
			}
			if (GlobalResidualSampleCount() == 0) {
				LOG(WARNING) << "[WeightedTSDF] Image " << imageID
					<< " has no reliable alignment window and no global P75; "
					<< "skipping integration.";
				integratedImages_.insert(imageID);
				integratedFrames = integratedFrameCount_;
				vertices = static_cast<int>(lastVertexCount_);
				triangles = static_cast<int>(lastTriangleCount_);
				return false;
			}

			const float globalResidualReference = std::max(
				CurrentGlobalResidualP75(), std::numeric_limits<float>::epsilon());
			const float alpha = static_cast<float>(
				std::max(weighted.UnsupportedRegionScale, 1e-6));
			const float unsupportedResidual = globalResidualReference / alpha;
			const size_t count = selectedQuality->alignmentWeightDenominator.size();
			std::vector<float> globalFusionMetrics(count, unsupportedResidual);
			for (size_t i = 0; i < count; ++i) {
				const float denom = selectedQuality->alignmentWeightDenominator[i];
				if (denom > 1e-6f) {
					globalFusionMetrics[i] = (
						selectedQuality->alignmentMetricNumerator[i] +
						unsupportedResidual * selectedQuality->alignmentFallbackWeight[i]) /
						denom;
				}
			}
			if (weighted.SaveQualityMaps) {
				try {
					SaveAlignmentWindowMaps(
						*selectedQuality, globalFusionMetrics,
						globalResidualReference, depthPath);
				}
				catch (const std::exception& e) {
					LOG(WARNING) << "[WeightedTSDF] Failed to save alignment-window maps for image "
						<< imageID << ": " << e.what();
				}
			}
			AppendAlignmentWindowDiagnostics(
				*selectedQuality, imageID, FileStem(depthPath),
				globalResidualReference);

			if (runtimeStats) {
				runtimeStats->globalResidualReference = globalResidualReference;
				runtimeStats->globalResidualReferenceQuantile =
					globalReferenceQuantile;
				runtimeStats->globalResidualSampleCount =
					GlobalResidualSampleCount();
			}

			open3d::t::geometry::Image upgradeEligibility = MakeTensorFloatMapImage(
				selectedQuality->upgradeEligibilityValues,
				selectedQuality->width,
				selectedQuality->height,
				"quality upgrade eligibility",
				device_);
			open3d::t::geometry::Image historyEligibility = MakeTensorFloatMapImage(
				selectedQuality->historyEligibilityValues,
				selectedQuality->width,
				selectedQuality->height,
				"history eligibility",
				device_);
			open3d::t::geometry::Image fusionMetric = MakeTensorFloatMapImage(
				globalFusionMetrics,
				selectedQuality->width,
				selectedQuality->height,
				"fusion metric",
				device_);
			const open3d::core::Tensor statistics = volume_->Integrate(
				blocks, depth, color, fusionMetric,
				upgradeEligibility, historyEligibility,
				intrinsic, intrinsic, extrinsic,
				depthScale, depthMax, truncVoxelMultiplier,
				globalResidualReference,
				static_cast<float>(weighted.HardRejectRatio),
				weighted.SoftSuppressionEnable,
				static_cast<float>(weighted.SoftSuppressionExponent),
				static_cast<float>(weighted.MaxVoxelWeight),
				static_cast<float>(
					weighted.ExtractionWeightThreshold /
					weighted.UpgradeCapDivisor));
			const std::vector<uint64_t> counts = statistics
				.To(open3d::core::Device("CPU:0"), true)
				.ToFlatVector<uint64_t>();
			if (runtimeStats && counts.size() >= 6) {
				const double candidates = static_cast<double>(counts[0]);
				const double accepted = static_cast<double>(counts[1]);
				runtimeStats->weighted = true;
				runtimeStats->globalP75Metric = weighted.GlobalP75Metric;
				runtimeStats->minUpgradeSamples = weighted.MinUpgradeSamples;
				runtimeStats->extractionWeightThreshold =
					weighted.ExtractionWeightThreshold;
				runtimeStats->upgradeCapDivisor = weighted.UpgradeCapDivisor;
				runtimeStats->upgradeWeightCap =
					weighted.ExtractionWeightThreshold / weighted.UpgradeCapDivisor;
				runtimeStats->hardRejectRatio =
					candidates > 0.0 ? counts[2] / candidates : 0.0;
				runtimeStats->softSuppressionRatio =
					accepted > 0.0 ? counts[3] / accepted : 0.0;
				runtimeStats->upgradeRatio =
					candidates > 0.0 ? counts[4] / candidates : 0.0;
				runtimeStats->meanEffectiveWeight =
					accepted > 0.0 ? counts[5] / accepted / 65535.0 : 0.0;
				if (counts.size() >= 8) {
					runtimeStats->upgradeEligibleRatio = candidates > 0.0
						? counts[6] / candidates : 0.0;
					runtimeStats->upgradeBlockedLowSamplesRatio = candidates > 0.0
						? counts[7] / candidates : 0.0;
				}
				if (counts.size() >= 14) {
					runtimeStats->historyUpdateEligibleRatio = candidates > 0.0
						? counts[8] / candidates : 0.0;
					runtimeStats->historyUpdateExcludedRejectRatio = candidates > 0.0
						? counts[9] / candidates : 0.0;
					runtimeStats->historyUpdateExcludedUnsupportedRatio = candidates > 0.0
						? counts[10] / candidates : 0.0;
					runtimeStats->upgradeBoostRatio = candidates > 0.0
						? counts[11] / candidates : 0.0;
					runtimeStats->meanUpgradeMultiplier = counts[11] > 0
						? counts[12] / static_cast<double>(counts[11]) / 65535.0
						: 0.0;
					runtimeStats->maxEffectiveWeight = counts[13] / 65535.0;
				}
			}
		}
		else {
			volume_->Integrate(
				blocks,
				depth,
				color,
				intrinsic,
				intrinsic,
				extrinsic,
				depthScale,
				depthMax);
		}
		if (runtimeStats) {
			runtimeStats->integrateMs =
				static_cast<double>(integrateTimer.ElapsedMicroSeconds()) / 1000.0;
			runtimeStats->weightedIntegrateMs = weighted.Enable
				? runtimeStats->integrateMs
				: 0.0;
		}
		integratedImages_.insert(imageID);
		++integratedFrameCount_;

		const int extractEvery = std::max(1, Config::getInstance().denseFusion.ExtractEveryNFrames);
		if (integratedFrameCount_ % extractEvery == 0) {
			Timer extractTimer;
			extractTimer.Start();
			Extract(renderData);
			if (runtimeStats) {
				runtimeStats->extractMs =
					static_cast<double>(extractTimer.ElapsedMicroSeconds()) / 1000.0;
				runtimeStats->displayPoints = renderData.dense_points.size();
				runtimeStats->meshVertices = lastVertexCount_;
				runtimeStats->meshTriangles = lastTriangleCount_;
			}
			renderUpdated = true;
		}

		integratedFrames = integratedFrameCount_;
		vertices = static_cast<int>(lastVertexCount_);
		triangles = static_cast<int>(lastTriangleCount_);
		if (runtimeStats) {
			runtimeStats->totalMs =
				static_cast<double>(totalTimer.ElapsedMicroSeconds()) / 1000.0;
			if (!renderUpdated) {
				runtimeStats->displayPoints = renderData.dense_points.size();
				runtimeStats->meshVertices = lastVertexCount_;
				runtimeStats->meshTriangles = lastTriangleCount_;
			}
		}
		return true;
	}

private:
	void ResetVolume()
	{
		globalResidualLower_ = std::priority_queue<float>();
		globalResidualUpper_ = std::priority_queue<
			float, std::vector<float>, std::greater<float>>();
		const auto& config = Config::getInstance().denseFusion;
		device_ = open3d::core::Device("CPU:0");
		if (ToLower(config.Device) == "cuda") {
			const open3d::core::Device cudaDevice("CUDA:0");
			if (cudaDevice.IsAvailable()) {
				device_ = cudaDevice;
			}
			else {
				LOG(WARNING) << "[DenseFusion] DenseFusion.Device=CUDA requested, "
					<< "but Open3D CUDA is not available. Falling back to CPU.";
			}
		}
		LOG(INFO) << "[DenseFusion] TSDF fusion backend: VoxelBlockGrid, device: "
			<< device_.ToString() << ".";
		LOG(INFO) << "[DenseFusion] TSDF tensors: depth/color=" << device_.ToString()
			<< ", intrinsic/extrinsic=CPU:0.";
		std::vector<std::string> attributeNames{ "tsdf", "weight", "color" };
		std::vector<open3d::core::Dtype> attributeTypes{
			open3d::core::Dtype::Float32,
			open3d::core::Dtype::Float32,
			open3d::core::Dtype::Float32 };
		std::vector<open3d::core::SizeVector> attributeChannels{
			{ 1 }, { 1 }, { 3 } };
		if (Config::getInstance().weightedTSDF.Enable) {
			attributeNames.push_back("residual_min");
			attributeTypes.push_back(open3d::core::Dtype::Float32);
			attributeChannels.push_back({ 1 });
			attributeNames.push_back("residual_mean");
			attributeTypes.push_back(open3d::core::Dtype::Float32);
			attributeChannels.push_back({ 1 });
			attributeNames.push_back("residual_count");
			attributeTypes.push_back(open3d::core::Dtype::UInt16);
			attributeChannels.push_back({ 1 });
		}
		volume_ = std::make_unique<open3d::t::geometry::VoxelBlockGrid>(
			attributeNames,
			attributeTypes,
			attributeChannels,
			static_cast<float>(config.VoxelLength),
			16,
			50000,
			device_);
	}

	void AddGlobalResidualMetric(const float value)
	{
		constexpr double quantile = 0.75;
		if (globalResidualLower_.empty() || value <= globalResidualLower_.top()) {
			globalResidualLower_.push(value);
		}
		else {
			globalResidualUpper_.push(value);
		}
		const size_t total = GlobalResidualSampleCount();
		const size_t desiredLowerSize = static_cast<size_t>(
			std::floor((total - 1) * quantile)) + 1;
		while (globalResidualLower_.size() > desiredLowerSize) {
			globalResidualUpper_.push(globalResidualLower_.top());
			globalResidualLower_.pop();
		}
		while (globalResidualLower_.size() < desiredLowerSize &&
			!globalResidualUpper_.empty()) {
			globalResidualLower_.push(globalResidualUpper_.top());
			globalResidualUpper_.pop();
		}
	}

	float CurrentGlobalResidualP75() const
	{
		constexpr double quantile = 0.75;
		if (globalResidualLower_.empty()) {
			return 0.0f;
		}
		const size_t total = GlobalResidualSampleCount();
		const double rank = (total - 1) * quantile;
		const double fraction = rank - std::floor(rank);
		const float lowerValue = globalResidualLower_.top();
		if (fraction <= 0.0 || globalResidualUpper_.empty()) {
			return lowerValue;
		}
		return static_cast<float>(
			lowerValue + fraction * (globalResidualUpper_.top() - lowerValue));
	}

	size_t GlobalResidualSampleCount() const
	{
		return globalResidualLower_.size() + globalResidualUpper_.size();
	}

	float ExtractionWeightThreshold() const
	{
		const auto& config = Config::getInstance();
		return config.weightedTSDF.Enable
			? static_cast<float>(config.weightedTSDF.ExtractionWeightThreshold)
			: 3.0f;
	}

	void Extract(RenderData& renderData)
	{
		renderData.dense_points.clear();
		renderData.dense_mesh_data.clear();
		renderData.dense_mesh_lines.clear();

		if (GenerateDensePoints()) {
			open3d::geometry::PointCloud pointCloud =
				volume_->ExtractPointCloud(ExtractionWeightThreshold()).ToLegacy();
			if (!pointCloud.IsEmpty()) {
				const size_t totalPoints = pointCloud.points_.size();
				const size_t maxDisplayPoints =
					std::max<size_t>(1, Config::getInstance().denseFusion.MaxDisplayPoints);
				const size_t stride = totalPoints > maxDisplayPoints
					? (totalPoints + maxDisplayPoints - 1) / maxDisplayPoints
					: 1;
				const size_t displayPoints = (totalPoints + stride - 1) / stride;
				renderData.dense_points.reserve(std::min(displayPoints, maxDisplayPoints));
				for (size_t i = 0;
					i < pointCloud.points_.size() && renderData.dense_points.size() < maxDisplayPoints;
					i += stride) {
					const Eigen::Vector3d color = i < pointCloud.colors_.size()
						? pointCloud.colors_[i]
						: DenseMeshWhiteColor();
					renderData.dense_points.push_back(MakePointData(pointCloud.points_[i], color, 1.0f));
				}
				if (stride > 1) {
					LOG(INFO) << "[DenseFusion] Display point cloud sampled: total="
						<< totalPoints
						<< ", display=" << renderData.dense_points.size()
						<< ", max=" << maxDisplayPoints
						<< ", stride=" << stride;
				}
			}
		}

		if (GenerateDenseMesh()) {
			open3d::geometry::TriangleMesh mesh =
				volume_->ExtractTriangleMesh(ExtractionWeightThreshold()).ToLegacy();
			if (mesh.IsEmpty()) {
				lastVertexCount_ = 0;
				lastTriangleCount_ = 0;
				return;
			}
			mesh.ComputeVertexNormals();
			lastVertexCount_ = mesh.vertices_.size();
			lastTriangleCount_ = mesh.triangles_.size();

			renderData.dense_mesh_data.reserve(mesh.triangles_.size());
			std::set<std::pair<int, int>> edges;
			const bool useImageColor = UseImageColor();
			for (const Eigen::Vector3i& tri : mesh.triangles_) {
				const auto make_vertex = [&](const int idx) {
					const Eigen::Vector3d color = useImageColor && idx < static_cast<int>(mesh.vertex_colors_.size())
						? mesh.vertex_colors_[idx]
						: DenseMeshWhiteColor();
					return MakePointData(mesh.vertices_[idx], color, 0.95f);
				};

				const PointPainterData a = make_vertex(tri.x());
				const PointPainterData b = make_vertex(tri.y());
				const PointPainterData c = make_vertex(tri.z());
				renderData.dense_mesh_data.push_back(MakeTriangleData(a, b, c));

				const int ids[3] = { tri.x(), tri.y(), tri.z() };
				for (int e = 0; e < 3; ++e) {
					int u = ids[e];
					int v = ids[(e + 1) % 3];
					if (u > v) {
						std::swap(u, v);
					}
					if (edges.insert({ u, v }).second) {
						renderData.dense_mesh_lines.push_back(
							MakeLineData(make_vertex(u), make_vertex(v)));
					}
				}
			}
		}
		else {
			lastVertexCount_ = 0;
			lastTriangleCount_ = 0;
		}
	}

public:
	bool ExportFullPly(const std::string& denseDir)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (!volume_ || integratedFrameCount_ == 0) {
			return false;
		}

		std::filesystem::create_directories(denseDir);
		const std::filesystem::path outputDir(denseDir);
		bool saved = false;
		LOG(INFO) << "[DenseFusion] Exporting full TSDF result on CPU.";

		open3d::t::geometry::VoxelBlockGrid exportVolume;
		try {
			exportVolume = volume_->To(open3d::core::Device("CPU:0"), true);
		}
		catch (const std::exception& e) {
			LOG(WARNING) << "[DenseFusion] Failed to copy TSDF volume to CPU for export: "
				<< e.what();
			return false;
		}

		try {
			const std::string tsdfVolumePath =
				NormalizePathSeparators((outputDir / "tsdf_volume.npz").string());
			exportVolume.Save(tsdfVolumePath);
			LOG(INFO) << "[DenseFusion] TSDF volume saved to " << tsdfVolumePath;
		}
		catch (const std::exception& e) {
			LOG(WARNING) << "[DenseFusion] Failed to save TSDF volume: "
				<< e.what();
		}

		try {
			open3d::geometry::PointCloud pointCloud =
				exportVolume.ExtractPointCloud(ExtractionWeightThreshold()).ToLegacy();
			const std::string pointCloudPath =
				NormalizePathSeparators((outputDir / "dense_cloud.ply").string());
			if (WritePointCloudPly(pointCloud, pointCloudPath)) {
				LOG(INFO) << "Dense point cloud saved to " << pointCloudPath;
				saved = true;
			}
		}
		catch (const std::exception& e) {
			LOG(WARNING) << "[DenseFusion] Failed to export dense point cloud from CPU TSDF: "
				<< e.what();
			return false;
		}

		try {
			open3d::geometry::TriangleMesh mesh =
				exportVolume.ExtractTriangleMesh(ExtractionWeightThreshold()).ToLegacy();
			const std::string meshPath =
				NormalizePathSeparators((outputDir / "dense_mesh.ply").string());
			if (WriteTriangleMeshPly(std::move(mesh), meshPath)) {
				LOG(INFO) << "Dense mesh saved to " << meshPath;
				saved = true;
			}
		}
		catch (const std::exception& e) {
			LOG(WARNING) << "[DenseFusion] Failed to export dense mesh from CPU TSDF: "
				<< e.what();
		}
		return saved;
	}

private:
	std::mutex mutex_;
	open3d::core::Device device_ = open3d::core::Device("CPU:0");
	std::unique_ptr<open3d::t::geometry::VoxelBlockGrid> volume_;
	std::set<image_t> integratedImages_;
	std::priority_queue<float> globalResidualLower_;
	std::priority_queue<float, std::vector<float>, std::greater<float>>
		globalResidualUpper_;
	int integratedFrameCount_ = 0;
	size_t lastVertexCount_ = 0;
	size_t lastTriangleCount_ = 0;
};

DenseTsdfFusion& Fusion()
{
	static DenseTsdfFusion fusion;
	return fusion;
}

} // namespace

bool ExportDenseFusionFullPly(const std::string& denseDir)
{
	return Fusion().ExportFullPly(denseDir);
}

void DenseFusionTask::run()
{
	if (!Config::getInstance().denseFusion.Enable) {
		return;
	}

	try {
		const std::string denseRoot = StringReplace(
			EnsureTrailingSlash(GetParentDir(GetParentDir(depthPath.toStdString()))),
			"\\",
			"/");
		GpuDiagnostics::SetOutputRoot(denseRoot);
		GpuDiagnostics::RecordEvent({
			"DenseFusion.Begin",
			static_cast<int>(imageID),
			"DenseFusion",
			depthPath.toStdString()
			});
		{
			QReadLocker locker(&(m_data->m_models_lock[0]));
			if (m_data->getModelManager()->Size() == 0) {
				throw std::runtime_error("No reconstruction is available for dense TSDF fusion.");
			}

			const std::shared_ptr<Reconstruction>& model = m_data->getModelManager()->Get(0);
			if (!model || !model->ExistsImage(imageID) || !model->IsImageRegistered(imageID)) {
				throw std::runtime_error("Image is not registered in model 0.");
			}
		}

		std::shared_ptr<RenderData> renderData = std::make_shared<RenderData>();

		int integratedFrames = 0;
		int vertices = 0;
		int triangles = 0;
		bool denseRenderUpdated = false;
		TsdfRuntimeStats runtimeStats;
		const std::string sourceImagePath = m_data->getImagePath(imageID);
		const std::shared_ptr<const AlignmentQualityData> qualityData =
			Config::getInstance().weightedTSDF.Enable
			? m_data->TakeAlignmentQuality(imageID)
			: nullptr;
		const bool integrated = Fusion().Integrate(
			imageID,
			depthPath.toStdString(),
			sourceImagePath,
			*renderData,
			integratedFrames,
			vertices,
			triangles,
			denseRenderUpdated,
			qualityData,
			&runtimeStats);

		if (denseRenderUpdated) {
			const size_t pointBytes =
				renderData->dense_points.size() * 7 * sizeof(float);
			const size_t meshBytes =
				renderData->dense_mesh_data.size() * 21 * sizeof(float) +
				renderData->dense_mesh_lines.size() * 14 * sizeof(float);
			GpuDiagnostics::RecordEvent({
				"DenseFusion.RenderDataReady",
				static_cast<int>(imageID),
				"DenseFusion",
				denseRenderUpdated ? "updated" : "not_updated",
				renderData->dense_points.size(),
				renderData->dense_mesh_data.size(),
				pointBytes + meshBytes
				});
			m_data->SetDenseRenderData(renderData);
		}
		if (integrated) {
			if (Config::getInstance().pipeline.OutputTimingFiles) {
				const auto processMemory = GpuDiagnostics::CurrentProcessMemory();
				const auto& denseConfig = Config::getInstance().denseFusion;
				const std::string header =
					"timestamp,image_id,voxel_length,trunc_distance,depth_max,"
					"valid_depth_pixels,integrate_ms,extract_ms,total_ms,"
					"integrated_frames,display_points,mesh_vertices,mesh_triangles,"
					"render_updated,working_set_mb,peak_working_set_mb,private_usage_mb,"
					"weighted,global_p75_metric,mean_effective_weight,hard_reject_ratio,"
					"soft_suppression_ratio,quality_upgrade_ratio,"
					"weighted_integration_ms,global_residual_reference,"
					"global_residual_reference_quantile,"
					"global_residual_sample_count,min_quality_upgrade_samples,"
					"upgrade_eligible_ratio,upgrade_blocked_low_samples_ratio,"
					"history_update_eligible_ratio,"
					"history_update_excluded_reject_ratio,"
					"history_update_excluded_unsupported_ratio,"
					"upgrade_boost_ratio,mean_upgrade_multiplier,"
					"max_effective_weight,upgrade_weight_cap_divisor,"
					"upgrade_weight_cap,"
					"extraction_weight_threshold";
				std::ostringstream row;
				row << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
					<< imageID << ","
					<< denseConfig.VoxelLength << ","
					<< denseConfig.VoxelLength * denseConfig.TruncVoxelMultiplier << ","
					<< denseConfig.DepthMax << ","
					<< runtimeStats.validDepthPixels << ","
					<< runtimeStats.integrateMs << ","
					<< runtimeStats.extractMs << ","
					<< runtimeStats.totalMs << ","
					<< integratedFrames << ","
					<< runtimeStats.displayPoints << ","
					<< runtimeStats.meshVertices << ","
					<< runtimeStats.meshTriangles << ","
					<< (denseRenderUpdated ? "true" : "false") << ","
					<< processMemory.workingSetMb << ","
					<< processMemory.peakWorkingSetMb << ","
					<< processMemory.privateUsageMb << ","
					<< (runtimeStats.weighted ? "true" : "false") << ","
					<< GpuDiagnostics::EscapeCsvValue(runtimeStats.globalP75Metric) << ","
					<< runtimeStats.meanEffectiveWeight << ","
					<< runtimeStats.hardRejectRatio << ","
					<< runtimeStats.softSuppressionRatio << ","
					<< runtimeStats.upgradeRatio << ","
					<< runtimeStats.weightedIntegrateMs << ","
					<< runtimeStats.globalResidualReference << ","
					<< runtimeStats.globalResidualReferenceQuantile << ","
					<< runtimeStats.globalResidualSampleCount << ","
					<< runtimeStats.minUpgradeSamples << ","
					<< runtimeStats.upgradeEligibleRatio << ","
					<< runtimeStats.upgradeBlockedLowSamplesRatio << ","
					<< runtimeStats.historyUpdateEligibleRatio << ","
					<< runtimeStats.historyUpdateExcludedRejectRatio << ","
					<< runtimeStats.historyUpdateExcludedUnsupportedRatio << ","
					<< runtimeStats.upgradeBoostRatio << ","
					<< runtimeStats.meanUpgradeMultiplier << ","
					<< runtimeStats.maxEffectiveWeight << ","
					<< runtimeStats.upgradeCapDivisor << ","
					<< runtimeStats.upgradeWeightCap << ","
					<< runtimeStats.extractionWeightThreshold;
				GpuDiagnostics::AppendTimingRow(
					"tsdf_runtime_stats.csv", header, row.str());

				const std::string frameHeader =
					"timestamp,image_id,image_name,global_feature_ms,retrieval_ms,"
					"sift_extract_ms,matching_ms,pose_estimation_ms,triangulation_ms,"
					"local_ba_ms,sfm_total_ms,depth_total_ms,depth_alignment_ms,"
					"tsdf_integrate_ms,tsdf_extract_ms,frame_elapsed_ms";
				std::ostringstream frameRow;
				frameRow << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
					<< imageID << ","
					<< GpuDiagnostics::EscapeCsvValue(FileStem(depthPath.toStdString())) << ","
					<< ",,,,,,,,,,"
					<< runtimeStats.integrateMs << ","
					<< runtimeStats.extractMs << ",";
				GpuDiagnostics::AppendTimingRow(
					"pipeline_frame_timing.csv", frameHeader, frameRow.str());
			}
			GpuDiagnostics::RecordEvent({
				"DenseFusion.End",
				static_cast<int>(imageID),
				"DenseFusion",
				denseRenderUpdated ? "render_updated" : "integrated_only",
				static_cast<size_t>(std::max(0, vertices)),
				static_cast<size_t>(std::max(0, triangles)),
				0
				});
			LOG(INFO) << "[DenseFusion] Integrated image " << imageID
				<< ", frames: " << integratedFrames
				<< ", vertices: " << vertices
				<< ", triangles: " << triangles
				<< ", dense render updated: "
				<< (denseRenderUpdated ? "true" : "false");
			emit denseFusionFinished(
				static_cast<quint32>(imageID),
				integratedFrames,
				vertices,
				triangles,
				denseRenderUpdated);
		}
		else {
			LOG(WARNING) << "[DenseFusion] Skipped image " << imageID
				<< ", frames: " << integratedFrames
				<< ", vertices: " << vertices
				<< ", triangles: " << triangles;
		}
	}
	catch (const std::exception& e) {
		const QString message = QString::fromStdString(e.what());
		LOG(ERROR) << "[DenseFusion] Image " << imageID << " failed: " << e.what();
		emit denseFusionFailed(static_cast<quint32>(imageID), message);
	}
}
