#include "Database.h"
#include "IO.h"
#include "../Base/config.h"

const size_t Database::kMaxNumImages = static_cast<size_t>(std::numeric_limits<int32_t>::max());
std::string Database::imageDir = "";
std::string Database::exportPath = "";

Database::Database()
{
	cameras.clear();
	images.clear();
	keypoints.clear();
	descriptors.clear();
	matches.clear();
	twoViewGeometries.clear();

	cameras.reserve(50);
	images.reserve(1500);
	keypoints.reserve(1500);
	descriptors.reserve(1500);
	matches.reserve(1124250);
	twoViewGeometries.reserve(1124250);

	// 根据配置文件读取全局特征模型类型，动态设置特征维度
	Config& config = Config::getInstance();
	std::string featureType = config.globalFeature.GlobalFeatureType;

	if (featureType == "VGG") {
		d = 2048;
		LOG(INFO) << "Using VGG with NetVLAD global feature extractor (2048-dim)";
	}
	else if (featureType == "SupScene" || featureType == "Supscene") {
		d = 1536;
		LOG(INFO) << "Using SupScene global feature extractor (1536-dim)";
	}	
	else {
		d = 2048;
		LOG(WARNING) << "Unknown GlobalFeatureType: " << featureType << ", defaulting to NetVLAD (2048-dim)";
	}

	std::lock_guard<std::mutex> lock(faissMu);
	indexHNSW = std::make_unique<faiss::IndexHNSWFlat>(d, M);
	//indexHNSW.hnsw.efConstruction = efConstruction;

}
Database::~Database()
{
	cameras.clear();
	images.clear();
	keypoints.clear();
	descriptors.clear();
	matches.clear();
	twoViewGeometries.clear();
}
bool Database::ExistsCamera(camera_t camera_id) const
{
	return cameras.find(camera_id) != cameras.end();
}
bool Database::ExistsCamera(const Camera& camera, camera_t& cameraID)const
{
	for (const auto& pair : cameras)
	{
		if (pair.second.Width() == camera.Width() && pair.second.Height() == camera.Height() && pair.second.Params() == camera.Params() && pair.second.ModelId() == camera.ModelId())
		{
			cameraID = pair.first;
			return true;
		}
	}
	cameraID = std::numeric_limits<camera_t>::max();
	return false;
}
bool Database::ExistsImage(image_t image_id) const
{
	return images.find(image_id) != images.end();
}
bool Database::ExistsImageWithName(const std::string& name) const
{
	for (const auto& pair : images)
	{
		if (pair.second.Name() == name)
		{
			return true;
		}
	}
	return false;
}
bool Database::ExistsKeypoints(image_t image_id) const
{
	return keypoints.find(image_id) != keypoints.end() && !keypoints.at(image_id).empty();
}
bool Database::ExistsGlobalFeature(image_t image_id) const
{
	std::lock_guard<std::mutex> lock(faissMu);
	if (!globalFeatureIndexHealthy) throw std::runtime_error("Global feature index is unavailable; reset the project");
	return globalFeaturePositions.count(image_id) != 0;
}
bool Database::ExistsDescriptors(image_t image_id) const
{
	return descriptors.find(image_id) != descriptors.end() && descriptors.at(image_id).rows() > 0;
}
bool Database::ExistsMatches(image_t image_id1, image_t image_id2) const
{
	const image_pair_t imagePairID = ImagePairToPairId(image_id1, image_id2);
	return matches.find(imagePairID) != matches.end() && !matches.at(imagePairID).empty();
}
bool Database::ExistsInlierMatches(image_t image_id1, image_t image_id2) const
{
	const image_pair_t imagePairID = ImagePairToPairId(image_id1, image_id2);
	return twoViewGeometries.find(imagePairID) != twoViewGeometries.end() && !twoViewGeometries.at(imagePairID).inlier_matches.empty();
}
size_t Database::NumCameras() const
{
	return cameras.size();
}
size_t Database::NumImages() const
{
	return images.size();
}
size_t Database::NumKeypointsForImage(image_t image_id) const
{
	return keypoints.at(image_id).size();
}
size_t Database::NumDescriptorsForImage(image_t image_id) const
{
	return descriptors.at(image_id).rows();
}
size_t Database::NumMatches() const
{
	size_t num = 0;
	for (const auto& pair : matches)
	{
		num += pair.second.size();
	}
	return num;
}
size_t Database::NumInlierMatches() const
{
	size_t num = 0;
	for (const auto& pair : twoViewGeometries)
	{
		num += pair.second.inlier_matches.size();
	}
	return num;
}
size_t Database::NumMatchedImagePairs() const
{
	return matches.size();
}
size_t Database::NumVerifiedImagePairs() const
{
	return twoViewGeometries.size();
}
Camera Database::ReadCamera(camera_t camera_id) const
{
	return cameras.at(camera_id);
}
std::vector<Camera> Database::ReadAllCameras() const
{
	std::vector<Camera> result;
	result.reserve(cameras.size());
	for (const auto& pair : cameras)
	{
		result.push_back(pair.second);
	}
	return result;
}
Image Database::ReadImage(image_t image_id) const
{
	return images.at(image_id);
}
Image Database::ReadImageWithName(const std::string& name) const
{
	for (const auto& pair : images)
	{
		if (pair.second.Name() == name)
		{
			return pair.second;
		}
	}
	std::cout << "ReadImageWithName error!" << std::endl;
	throw "Error!";
}
std::vector<Image> Database::ReadAllImages() const
{
	std::vector<Image> result;
	result.reserve(images.size());

	for (const auto& pair : images)
	{
		result.push_back(pair.second);
	}
	return result;
}
FeatureKeypoints Database::ReadKeypoints(image_t image_id) const
{
	return keypoints.at(image_id);
}
FeatureDescriptors Database::ReadDescriptors(image_t image_id) const
{
	return descriptors.at(image_id);
}
GlobalFeature Database::ReadGlobalFeature(image_t image_id) const
{
	std::lock_guard<std::mutex> lock(faissMu);
	if (!globalFeatureIndexHealthy) throw std::runtime_error("Global feature index is unavailable; reset the project");
	const auto it = globalFeaturePositions.find(image_id);
	if (it == globalFeaturePositions.end()) {
		throw std::runtime_error("Missing global feature for image " + std::to_string(image_id));
	}
	return globalFeatures.at(it->second);
}
FeatureMatches Database::ReadMatches(image_t image_id1, image_t image_id2) const
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
	FeatureMatches result = matches.at(pair_id);
	if (SwapImagePair(image_id1, image_id2))
	{
		for (FeatureMatch& match : result)
		{
			std::swap(match.point2D_idx1, match.point2D_idx2);
		}
	}
	return result;
}
std::vector<std::pair<image_pair_t, FeatureMatches>> Database::ReadAllMatches() const
{
	std::vector<std::pair<image_pair_t, FeatureMatches>> result;
	result.reserve(matches.size());

	for (const auto& pair : matches)
	{
		result.push_back(pair);
	}
	return result;
}
TwoViewGeometry Database::ReadTwoViewGeometry(image_t image_id1, image_t image_id2) const
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
	TwoViewGeometry result = twoViewGeometries.at(pair_id);
	if (SwapImagePair(image_id1, image_id2))
	{
		result.Invert();
	}
	return result;
}
void Database::ReadTwoViewGeometries(std::vector<image_pair_t>* image_pair_ids, std::vector<TwoViewGeometry>* two_view_geometries) const
{
	image_pair_ids->reserve(image_pair_ids->size() + twoViewGeometries.size());
	two_view_geometries->reserve(two_view_geometries->size() + twoViewGeometries.size());

	for (const auto& pair : twoViewGeometries)
	{
		image_pair_ids->push_back(pair.first);
		two_view_geometries->push_back(pair.second);
	}
}
void Database::ReadTwoViewGeometryNumInliers(std::vector<std::pair<image_t, image_t>>* image_pairs, std::vector<int>* num_inliers) const
{
	const auto num_inlier_matches = NumInlierMatches();
	image_pairs->reserve(image_pairs->size() + num_inlier_matches);
	num_inliers->reserve(num_inliers->size() + num_inlier_matches);

	for (const auto& pair : twoViewGeometries)
	{
		image_t image_id1;
		image_t image_id2;
		PairIdToImagePair(pair.first, &image_id1, &image_id2);
		image_pairs->push_back(std::make_pair(image_id1, image_id2));

		num_inliers->push_back(pair.second.inlier_matches.size());
	}
}
camera_t Database::WriteCamera(const Camera& camera, bool use_camera_id)
{
	if (use_camera_id)
	{
		CHECK(!ExistsCamera(camera.CameraId())) << "camera_id must be unique";
		cameras[camera.CameraId()] = camera;
		return camera.CameraId();
	}
	camera_t alreadyCameraID = std::numeric_limits<camera_t>::max();
	if (ExistsCamera(camera, alreadyCameraID))
	{
		return alreadyCameraID;
	}
	Camera backup = camera;
	camera_t tryCameraID = 1;
	while (ExistsCamera(tryCameraID))
	{
		tryCameraID++;
	}
	backup.SetCameraId(tryCameraID);
	cameras[tryCameraID] = backup;
	return tryCameraID;
}
image_t Database::WriteImage(const Image& image, bool use_image_id)
{
	if (use_image_id)
	{
		CHECK(!ExistsImage(image.ImageId())) << "image_id must be unique";
		images[image.ImageId()] = image;
		return image.ImageId();
	}

	Image backup = image;
	const image_t newImageID = images.size() + 1;
	backup.SetImageId(newImageID);
	images[newImageID] = backup;
	return newImageID;
}
void Database::WriteKeypoints(image_t image_id, const FeatureKeypoints& keypoints)
{
	this->keypoints[image_id] = keypoints;
}
void Database::WriteGlobalFeature(image_t image_id, const GlobalFeature& globalFeature)
{
	if (image_id == 0 || image_id >= kMaxNumImages || globalFeature.size() != d ||
		!std::all_of(globalFeature.begin(), globalFeature.end(), [](float v) { return std::isfinite(v); })) {
		throw std::invalid_argument("Invalid global feature for image " + std::to_string(image_id));
	}
	std::lock_guard<std::mutex> lock(faissMu);
	if (!globalFeatureIndexHealthy) throw std::runtime_error("Global feature index is unavailable; reset the project");
	const auto existing = globalFeaturePositions.find(image_id);
	if (existing != globalFeaturePositions.end()) {
		if (globalFeatures.at(existing->second) != globalFeature) {
			throw std::runtime_error("Conflicting global feature for image " + std::to_string(image_id));
		}
		return;
	}
	const size_t position = globalFeatures.size();
	try {
		globalFeatures.push_back(globalFeature);
		globalFeatureImageIds.push_back(image_id);
		globalFeaturePositions.emplace(image_id, position);
	} catch (...) {
		globalFeatures.resize(position);
		globalFeatureImageIds.resize(position);
		globalFeaturePositions.erase(image_id);
		throw;
	}
	try {
		indexHNSW->add(1, globalFeatures.back().data());
	} catch (...) {
		globalFeatures.resize(position);
		globalFeatureImageIds.resize(position);
		globalFeaturePositions.erase(image_id);
		globalFeatureIndexHealthy = false;
		throw;
	}
}
void Database::WriteDescriptors(image_t image_id, const FeatureDescriptors& descriptors)
{
	this->descriptors[image_id] = descriptors;
}
void Database::WriteMatches(image_t image_id1, image_t image_id2, const FeatureMatches& matches)
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
	if (SwapImagePair(image_id1, image_id2))
	{
		FeatureMatches backup = matches;
		for (FeatureMatch& match : backup)
		{
			std::swap(match.point2D_idx1, match.point2D_idx2);
		}
		this->matches[pair_id] = backup;
	}
	else
	{
		this->matches[pair_id] = matches;
	}
}
void Database::WriteTwoViewGeometry(image_t image_id1, image_t image_id2, const TwoViewGeometry& two_view_geometry)
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
	if (SwapImagePair(image_id1, image_id2))
	{
		TwoViewGeometry backup = two_view_geometry;
		backup.Invert();
		twoViewGeometries[pair_id] = backup;
	}
	else
	{
		twoViewGeometries[pair_id] = two_view_geometry;
	}
}
std::vector<image_t> Database::SearchGlobalFeatures(image_t query_image_id, size_t top_n) const
{
	if (top_n == 0) return {};
	std::lock_guard<std::mutex> lock(faissMu);
	if (!globalFeatureIndexHealthy) throw std::runtime_error("Global feature index is unavailable; reset the project");
	const auto it = globalFeaturePositions.find(query_image_id);
	if (it == globalFeaturePositions.end()) {
		throw std::runtime_error("Missing global feature for image " + std::to_string(query_image_id));
	}
	const size_t count = top_n >= globalFeatures.size() ? globalFeatures.size() : top_n + 1;
	std::vector<float> distances(count);
	std::vector<faiss::idx_t> labels(count);
	indexHNSW->search(1, globalFeatures.at(it->second).data(), count, distances.data(), labels.data());
	std::vector<image_t> result;
	result.reserve(std::min(top_n, count));
	for (const auto label : labels) {
		if (label < 0 || static_cast<size_t>(label) >= globalFeatureImageIds.size()) continue;
		const image_t id = globalFeatureImageIds[static_cast<size_t>(label)];
		if (id != query_image_id) result.push_back(id);
		if (result.size() == top_n) break;
	}
	return result;
}

void Database::ResetGlobalFeatures()
{
	std::lock_guard<std::mutex> lock(faissMu);
	auto emptyIndex = std::make_unique<faiss::IndexHNSWFlat>(d, M);
	globalFeatures.clear();
	globalFeaturePositions.clear();
	globalFeatureImageIds.clear();
	indexHNSW = std::move(emptyIndex);
	globalFeatureIndexHealthy = true;
}
void Database::UpdateCamera(const Camera& camera)
{
	cameras[camera.CameraId()] = camera;
}
void Database::UpdateImage(const Image& image)
{
	images[image.ImageId()] = image;
}
void Database::DeleteMatches(image_t image_id1, image_t image_id2)
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
#ifdef _DEBUG
	matches.erase(pair_id);
#else
	matches.erase(pair_id);
#endif
}
void Database::DeleteInlierMatches(image_t image_id1, image_t image_id2)
{
	const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);

#ifdef _DEBUG
	twoViewGeometries.erase(pair_id);
#else
	twoViewGeometries.erase(pair_id);
#endif
}
void Database::ClearAllTables()
{
	ResetGlobalFeatures();
	cameras.clear();
	images.clear();
	keypoints.clear();
	descriptors.clear();
	matches.clear();
	twoViewGeometries.clear();
}
void Database::ClearCameras()
{
	cameras.clear();
}
void Database::ClearImages()
{
	ResetGlobalFeatures();
	images.clear();
}
void Database::ClearDescriptors()
{
	descriptors.clear();
}
void Database::ClearKeypoints()
{
	keypoints.clear();
}
void Database::ClearMatches()
{
	matches.clear();
}
void Database::ClearTwoViewGeometries()
{
	twoViewGeometries.clear();
}
void Database::Merge(const Database& database1, const Database& database2, Database* merged_database)
{
	std::unordered_map<camera_t, camera_t> new_camera_ids1;
	for (const auto& camera : database1.ReadAllCameras())
	{
		const camera_t new_camera_id = merged_database->WriteCamera(camera);
		new_camera_ids1.emplace(camera.CameraId(), new_camera_id);
	}

	std::unordered_map<camera_t, camera_t> new_camera_ids2;
	for (const auto& camera : database2.ReadAllCameras())
	{
		const camera_t new_camera_id = merged_database->WriteCamera(camera);
		new_camera_ids2.emplace(camera.CameraId(), new_camera_id);
	}

	// Merge the images.

	std::unordered_map<image_t, image_t> new_image_ids1;
	for (auto& image : database1.ReadAllImages())
	{
		image.SetCameraId(new_camera_ids1.at(image.CameraId()));
		CHECK(!merged_database->ExistsImageWithName(image.Name()))
			<< "The two databases must not contain images with the same name, but "
			"the there are images with name "
			<< image.Name() << " in both databases";
		const image_t new_image_id = merged_database->WriteImage(image);
		new_image_ids1.emplace(image.ImageId(), new_image_id);
		const auto keypoints = database1.ReadKeypoints(image.ImageId());
		const auto descriptors = database1.ReadDescriptors(image.ImageId());
		merged_database->WriteKeypoints(new_image_id, keypoints);
		merged_database->WriteDescriptors(new_image_id, descriptors);
	}

	std::unordered_map<image_t, image_t> new_image_ids2;
	for (auto& image : database2.ReadAllImages())
	{
		image.SetCameraId(new_camera_ids2.at(image.CameraId()));
		CHECK(!merged_database->ExistsImageWithName(image.Name()))
			<< "The two databases must not contain images with the same name, but "
			"the there are images with name "
			<< image.Name() << " in both databases";
		const image_t new_image_id = merged_database->WriteImage(image);
		new_image_ids2.emplace(image.ImageId(), new_image_id);
		const auto keypoints = database2.ReadKeypoints(image.ImageId());
		const auto descriptors = database2.ReadDescriptors(image.ImageId());
		merged_database->WriteKeypoints(new_image_id, keypoints);
		merged_database->WriteDescriptors(new_image_id, descriptors);
	}

	// Merge the matches.

	for (const auto& matches : database1.ReadAllMatches())
	{
		image_t image_id1, image_id2;
		Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

		const image_t new_image_id1 = new_image_ids1.at(image_id1);
		const image_t new_image_id2 = new_image_ids1.at(image_id2);

		merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
	}

	for (const auto& matches : database2.ReadAllMatches())
	{
		image_t image_id1, image_id2;
		Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

		const image_t new_image_id1 = new_image_ids2.at(image_id1);
		const image_t new_image_id2 = new_image_ids2.at(image_id2);

		merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
	}

	// Merge the two-view geometries.

	{
		std::vector<image_pair_t> image_pair_ids;
		std::vector<TwoViewGeometry> two_view_geometries;
		database1.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

		for (size_t i = 0; i < two_view_geometries.size(); ++i) {
			image_t image_id1, image_id2;
			Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);

			const image_t new_image_id1 = new_image_ids1.at(image_id1);
			const image_t new_image_id2 = new_image_ids1.at(image_id2);

			merged_database->WriteTwoViewGeometry(
				new_image_id1, new_image_id2, two_view_geometries[i]);
		}
	}

	{
		std::vector<image_pair_t> image_pair_ids;
		std::vector<TwoViewGeometry> two_view_geometries;
		database2.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

		for (size_t i = 0; i < two_view_geometries.size(); ++i) {
			image_t image_id1, image_id2;
			Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);

			const image_t new_image_id1 = new_image_ids2.at(image_id1);
			const image_t new_image_id2 = new_image_ids2.at(image_id2);

			merged_database->WriteTwoViewGeometry(
				new_image_id1, new_image_id2, two_view_geometries[i]);
		}
	}
}
void Database::Export(const std::string& exportPath)
{
	DatabaseExporter exporter(this, exportPath);
	exporter.Export();
}
