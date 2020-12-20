from diy_gym.addons.addon import Addon
import numpy as np


class DebugMarkers(Addon):
    """
    Adds a debug market to frame
    
    config:
    - target_frame_name
    """
    def __init__(self, parent, config):
        super().__init__(parent, config)
        
        self.source_model = parent.models[config.get('source_model')]
        self.source_frame_id = self.source_model.get_frame_id(
            config.get('source_frame')) if 'source_frame' in config else -1
        
        color = config.get('color', [1, 0, 0])
        text_color = config.get('text_color', [.5, .1, 0])
        
        source_xyz = p.getLinkState(
            self.source_model.uid,
            self.source_frame_id)[4] if self.source_frame_id >= 0 else p.getBasePositionAndOrientation(
                self.source_model.uid)[0]
        
        if 'text' in config:
            p.addUserDebugText(
                text=config.get('text'), 
                textPosition=[0, 0, 0.1],
                textColorRGB=text_color,
                textSize=1.5,
                parentObjectUniqueId=self.source_model.uid,
                parentLinkIndex=self.source_frame_id
            )
            
        if 'target_model' in config:
            self.target_model = parent.models[config.get('target_model')]
            self.target_frame_id = self.target_model.get_frame_id(
            config.get('target_frame')) if 'target_frame' in config else -1
            target_xyz = p.getLinkState(
                self.target_model.uid,
                self.target_frame_id)[4] if self.target_frame_id >= 0 else p.getBasePositionAndOrientation(
                    self.target_model.uid)[0] 
            
            p.addUserDebugLine(source_xyz, target_xyz, lineColorRGB=color)
        else:   
            d = config.get('distance', 0.05)
            o = np.array(source_xyz)
            p.addUserDebugLine(o+[-d, 0, 0], o+[d, 0, 0], color, 
                               parentObjectUniqueId=self.source_model.uid, 
                               parentLinkIndex=self.source_frame_id)
            p.addUserDebugLine(o+[0, -d, 0], o+[0, d, 0], color, 
                               parentObjectUniqueId=self.source_model.uid, 
                               parentLinkIndex=self.source_frame_id)
            p.addUserDebugLine(o+[0, 0, -d], o+[0, 0, d], color, 
                               parentObjectUniqueId=self.source_model.uid, 
                               parentLinkIndex=self.source_frame_id)
