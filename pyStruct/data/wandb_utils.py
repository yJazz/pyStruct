import numpy as np 
import pickle
import wandb


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer( fig.canvas.tostring_argb(), dtype=np.uint8 )
    # buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (h, w,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def log_file_as_artifact(project_name, filepath, save_name, artifact_name, artifact_type):
    with wandb.init(project= project_name, job_type='load-data') as run:
        # init an artifact
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        # Add to the artifact
        artifact.add_file(filepath, save_name)
        run.log_artifact(artifact)

def log_pickle_as_artifact(project_name , object, save_name, artifact_name, artifact_type):
    with wandb.init(project= project_name, job_type='load-data') as run:
        # init an artifact
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        with artifact.new_file(save_name, 'wb') as f:
            pickle.dump(object, f)
        run.log_artifact(artifact)
    return