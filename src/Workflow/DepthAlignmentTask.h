#pragma once

#include "BaseTask.h"
#include "../Base/types.h"

#include <QString>
#include <string>
#include <vector>

struct AlignmentQualityData;

// Saves full-resolution maps after DenseFusion has determined the frame P75.
void SaveAlignmentWindowMaps(
	const AlignmentQualityData& quality,
	const std::vector<float>& fusionMetric,
	float globalReference,
	const std::string& alignedDepthPath);

void AppendAlignmentWindowDiagnostics(
	const AlignmentQualityData& quality,
	image_t imageID,
	const std::string& imageName,
	float globalReference);

class DepthAlignmentTask : public BaseTask
{
	Q_OBJECT

public:
	explicit DepthAlignmentTask(
		WorkData* data,
		image_t imageID,
		const QString& depthPath,
		const QString& colorPath,
		QObject* parent = nullptr);

	void run() override;

signals:
	void depthAlignmentFinished(quint32 imageID, QString alignedDepthPath, QString colorPath);
	void depthAlignmentFailed(quint32 imageID, QString errorMessage);

private:
	image_t imageID_;
	QString rawDepthPath_;
	QString colorPath_;
};
