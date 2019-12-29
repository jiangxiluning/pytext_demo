from pytext.task import NewTask
from .model import MyTagger
from .metric import MyTaggingMetricReporter


class MyTaggingTask(NewTask):
    class Config(NewTask.Config):
        model: MyTagger.Config = MyTagger.Config()
        metric_reporter: MyTaggingMetricReporter.Config = MyTaggingMetricReporter.Config()

