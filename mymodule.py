from keras.utils import io_utils

class SaveModelToGitHub(keras.callbacks.Callback):
  def __init__(self, github_settings, name = "model", only_save_at_end = False, delete_after_upload = False):
    super().__init__()
    self.github_settings = github_settings
    self.name = name
    self.only_save_at_end = only_save_at_end
    self.delete_after_upload = delete_after_upload

  def on_epoch_end(self, epoch, logs=None):
    if self.only_save_at_end == False or (self.only_save_at_end and self.params["epochs"] - 1 == epoch):
      self._save_model(epoch)

  def _save_model(self, epoch):
    filename = f"{self.name}-epoch-{epoch + 1}.h5"
    filepath = io_utils.path_to_string(filename)
    self.model.save(filepath)

    
