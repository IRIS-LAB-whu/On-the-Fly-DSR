#pragma once

#include "../Base/types.h"
#include "BaseTask.h"

#include <QString>

#include <string>

class DenseFusionTask : public BaseTask
{
	Q_OBJECT

public:
	explicit DenseFusionTask(
		WorkData* data,
		image_t imageID,
		const QString& depthPath,
		const QString& colorPath,
		QObject* parent = nullptr)
		: BaseTask(data, parent),
		imageID(imageID),
		depthPath(depthPath),
		colorPath(colorPath)
	{
	}

	void run() override;

signals:
	void denseFusionFinished(
		quint32 imageID,
		int integratedFrames,
		int vertices,
		int triangles,
		bool denseRenderUpdated);
	void denseFusionFailed(quint32 imageID, QString errorMessage);

private:
	image_t imageID;
	QString depthPath;
	QString colorPath;
};

bool ExportDenseFusionFullPly(const std::string& denseDir);
