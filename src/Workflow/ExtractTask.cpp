#include "../Base/timer.h"
#include "../Feature/Bitmap.h"
#include "../Base/config.h"
#include "ExtractTask.h"
#include "WorkData.h"

#include "glog/logging.h"

#include <memory>

void ExtractTask::run()
{
	std::string imagePathstr = imagePath.toStdString();
	try {
		m_data->setImageStatus(imagePathstr, CImageStatusFlag::CExtracting);
		Timer ExtractionTimer;
		ExtractionTimer.Start();
		CSIFTExtractionOptions options;
		options.maxNumFeatures = Config::getInstance().sfm.maxNumFeatures;
		LOG(INFO) << "[Extraction] SIFT max features: " << options.maxNumFeatures;
		std::unique_ptr<CSIFTGPUExtractor> SIFTGPUExtractor(new CSIFTGPUExtractor(options));
		FeatureKeypoints keypoints;
		FeatureDescriptors descriptors;
		LOG(INFO) << "[Extraction] Running sift extract: " << imagePathstr;
		Bitmap bitmap = m_data->ReadBitmap(imageid);
		if (SIFTGPUExtractor->Extract(bitmap, keypoints, descriptors))
		{
			const size_t imageID = m_data->getImageID(imagePathstr);
			const size_t cameraID = m_data->DatabaseReadImage(imageID).CameraId();
			ScaleKeypoints(bitmap, m_data->DatabaseReadCamera(cameraID), &keypoints);
			m_data->DatabaseWriteKeypoints(imageID, keypoints);
			m_data->DatabaseWriteDescriptors(imageID, descriptors);
			m_data->setImageStatus(imagePathstr, CImageStatusFlag::CExtracted);

			const size_t ExtractionTime = ExtractionTimer.ElapsedMicroSeconds() / 1000;
			m_data->SIFTExtractionTimes[imageID] += ExtractionTime;

			LOG(INFO) << "[Extraction] Image " << imagePathstr << " extracted successed!";

			m_data->ReleaseBitmap(imageid);
			emit localExtractFinished(imagePath);
		}
		else {
			emit frontEndFailed(imagePath, "local_extraction_failed");
		}
	}
	catch (const std::exception& e) {
		LOG(ERROR) << "[Extraction] Image " << imagePathstr << " failed: " << e.what();
		emit frontEndFailed(imagePath, QString::fromUtf8(e.what()));
	}
	catch (...) {
		LOG(ERROR) << "[Extraction] Image " << imagePathstr << " failed with unknown exception.";
		emit frontEndFailed(imagePath, "local_extraction_exception");
	}
}
