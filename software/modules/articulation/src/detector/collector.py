import lcm

from time import time
from sys import argv
from optparse import OptionParser
import threading
from collections import defaultdict, deque

from articulation.pose_list_msg_t import pose_list_msg_t
from articulation.track_msg_t import track_msg_t

# Articulation collector
# Main idea: collect live detections and compose
# tracks that are sent every f frames with p poses

class ArticulationCollector(object):
    def __init__(self, frames=10, poses=100): 

        self._poses = poses;
        self._frames = frames;
        self._tracks = defaultdict(lambda : deque(maxlen=self._poses))
        self._tracks_count = defaultdict(lambda : 0)
        self._channels = defaultdict(list)

        self._lc = lcm.LCM();
        self._sub = self._lc.subscribe("ARTICULATION_POSE_LIST", self.on_pose_list)

        self.running = True;

        
    def on_pose_list(self, channel, data): 
        poses = pose_list_msg_t.decode(data)

        # Receive and accumulate tracks
        for pose in poses.pose: 
            self._tracks[pose.id].append(pose)
            self._tracks_count[pose.id] +=1
            #print "Tracks %s length: %i" % (pose.id, len(self._tracks[pose.id]))

        # Send Accumulated tracks every k frames
        for tid,count in self._tracks_count.iteritems(): 
            if count % self._frames == 0: 
                print '=======> \n Publishing track %i with size: %i' % (tid,len(self._tracks[tid]))

                track_msg = track_msg_t()
                track_msg.id = tid;

                track_msg.pose = self._tracks[tid] # Leak?
                track_msg.pose_flags = [track_msg_t.POSE_VISIBLE for pose in track_msg.pose]
                track_msg.num_poses = len(self._tracks[tid])

                track_msg.pose_projected = None
                track_msg.num_poses_projected = 0;

                track_msg.pose_resampled = None
                track_msg.num_poses_resampled = 0;                
                
                track_msg.channels = None
                track_msg.num_channels = 0;
                
                # for name in ['width','height']:
                #     ch = channel_msg_t()
                #     ch.name = name
                    
                #     ch.values = [0 for pose in track_msg.pose]
                #     ch.num_values = len(ch.values)

                #     self._channels.append(ch)

                # track_msg.channels = self._channels
                # track_msg.num_channels = len(channels)
                # ch.name = "width"; 

                self._lc.publish("ARTICULATION_OBJECT_TRACKS", track_msg.encode())
        
    def run(self): 
        while self.running:
            self._lc.handle()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--frames", dest="frames", default=10, 
                      help="Send every f frames")

    parser.add_option("-p", "--poses", dest="poses", default=100, 
                       help="Send p poses")
    (options, args) = parser.parse_args()

    try: 
        # Articulation Collector
        bg = ArticulationCollector(frames=int(options.frames), poses=int(options.poses))
        bg.run()
    except KeyboardInterrupt: 
        bg.running = False

