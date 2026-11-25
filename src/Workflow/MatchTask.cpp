#include "../Base/timer.h"
#include "glog/logging.h"
#include "MatchTask.h"
#include "WorkData.h"

#include <memory>


void MatchTask::run()
{
	try {
		Timer SIFTMatchingTimer;
		SIFTMatchingTimer.Start();

		CSIFTMatchingOptions options;
		const FeatureDescriptors& descriptors1 = m_data->DatabaseReadDescriptors(imageID1);
		const FeatureDescriptors& descriptors2 = m_data->DatabaseReadDescriptors(imageID2);
		CHECK(descriptors1.cols() == 128 && descriptors2.cols() == 128);

		FeatureMatches matches;
		std::unique_ptr<CSIFTGPUMatcher> SIFTGPUMatcher(new CSIFTGPUMatcher(options));
		matches = SIFTGPUMatcher->Match(descriptors1, descriptors2);
		if (!matches.empty())
		{
			m_data->DatabaseWriteMatches(imageID1, imageID2, matches);
		}
		bool existsMatches = m_data->DatabaseExistsMatches(imageID1, imageID2);
		const size_t numMatches = (existsMatches ? m_data->DatabaseReadMatches(imageID1, imageID2).size() : 0);
		//LOG(INFO) << "[SIFTMatchGPU] " << imageID1 << " - " << imageID2 << ": " << numMatches;

		const size_t SIFTMatchingTime = SIFTMatchingTimer.ElapsedMicroSeconds() / 1000;
		m_data->SIFTMatchingTimes[imageID2] += SIFTMatchingTime;

		emit matchTaskFinished(imageID1, imageID2);
	}
	catch (const std::exception& e) {
		LOG(ERROR) << "[SIFTMatchGPU] " << imageID1 << " - " << imageID2
			<< " failed: " << e.what();
		emit matchingPairFailed(imageID1, imageID2, QString::fromUtf8(e.what()));
	}
	catch (...) {
		LOG(ERROR) << "[SIFTMatchGPU] " << imageID1 << " - " << imageID2
			<< " failed with unknown exception.";
		emit matchingPairFailed(imageID1, imageID2, "matching_exception");
	}
}
