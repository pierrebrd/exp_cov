# Importazione delle librerie necessarie
import rospy  # Libreria ROS per Python
import argparse  # Per il parsing degli argomenti da linea di comando
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  # Messaggi per la navigazione
from actionlib_msgs.msg import GoalStatus  # Stati dei goal di navigazione
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion  # Tipi geometrici ROS
from tf.transformations import quaternion_from_euler  # Conversione da angoli di Euler a quaternioni
from actionlib.action_client import ActionClient, CommState, get_name_of_constant  # Client per azioni ROS
from nav_msgs.msg import Odometry  # Messaggi di odometria
from nav_msgs.srv import GetPlan  # Servizio per ottenere il piano di navigazione

# Classe per gestire gli stati semplificati del goal
class SimpleGoalState:
    PENDING = 0  # In attesa di essere processato
    ACTIVE = 1   # In esecuzione
    DONE = 2     # Completato

# Aggiunge il metodo to_string alla classe SimpleGoalState
SimpleGoalState.to_string = classmethod(get_name_of_constant)

# Classe principale per l'invio dei waypoint
class waypoint_sender():

    def __init__(self, waypoints):
        """Inizializza il sender dei waypoint"""
        self.pose_seq = list()
        self.odom = None  # Inizializza odom a None
        # Crea un quaternione con orientamento neutro (0,0,0)
        x, y, z, w = quaternion_from_euler(0, 0, 0)
        # Converte i waypoint in pose complete (posizione + orientamento)
        for waypoint in waypoints:
            self.pose_seq.append(Pose(waypoint, Quaternion(x, y, z, w)))
        rospy.loginfo(f"Waypoint sender started. Will send up to {len(self.pose_seq)} waypoints.")
        self.goal_cnt = 0  # Contatore dei waypoint

        # Inizializza il client per move_base
        self.action_client = ActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        wait = self.action_client.wait_for_server()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return

        # Attende il servizio di pianificazione
        rospy.wait_for_service('/move_base/make_plan')
        self.get_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)
        
        # Sottoscrizione al topic di odometria e attesa dati
        self.odom_srv = rospy.Subscriber("/odom", Odometry, callback=self.save_odom)
        rospy.loginfo("Waiting for odometry data...")
        while self.odom is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Odometry data received")
        rospy.loginfo("Waypoint sender connected to move base server.")
        rospy.loginfo("Starting waypoint navigation.")

        # Configura i parametri di timeout e controllo
        self.MAX_WAIT = 120  # Timeout massimo per raggiungere un waypoint
        self.PLAN_CHECK_WAIT = 15  # Intervallo tra i controlli del piano
        self.PLAN_CHECK_MAX_RETRIES = 3  # Numero massimo di tentativi di pianificazione
        self.simple_state = SimpleGoalState.DONE
        
        # Invia il primo goal e mantiene il nodo attivo
        self.send_goal()
        rospy.spin()

    def save_odom(self, msg):
        """Salva l'ultima posizione del robot dall'odometria"""
        self.odom = msg.pose.pose

    def active_cb(self):
        """Callback chiamata quando un goal diventa attivo"""
        rospy.loginfo(f"Goal pose {self.goal_cnt} is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        """Gestisce il feedback durante l'esecuzione del goal"""
        rospy.loginfo(f"Feedback for goal pose {self.goal_cnt} received")
        
        # Verifica timeout
        if abs(rospy.Time.now().secs - self.goal_start) >= self.MAX_WAIT:
            if self.goal_cnt < len(self.pose_seq):
                rospy.loginfo(f"Timeout reached for current goal: {self.goal_cnt}. Robot position: x:{self.odom.position.x:.2f} y:{self.odom.position.y:.2f}")
                self.goal_cnt += 1
                self.send_goal()
            else:
                rospy.loginfo(f"Timeout reached for last goal: {self.goal_cnt}. Robot position: x:{self.odom.position.x:.2f} y:{self.odom.position.y:.2f}")
                rospy.signal_shutdown("Waypoint navigation ended.")
                return

        # Verifica periodica del piano
        elif abs(rospy.Time.now().secs - self.last_plan_check) > self.PLAN_CHECK_WAIT:
            # Prepara le pose di partenza e arrivo per la verifica
            start = PoseStamped()
            start.header.frame_id = "map"
            start.header.stamp = rospy.Time.now() 
            start.pose = self.odom
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now() 
            goal.pose = self.pose_seq[self.goal_cnt]
            tolerance = 0.5  # Tolleranza per il piano

            # Tenta di ottenere un piano
            no_plan = True
            try:
                plan = self.get_plan(start, goal, tolerance)
                no_plan = len(plan.plan.poses) == 0
            except Exception as e:
                no_plan = True
                rospy.loginfo(f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} failed with exception.")
                rospy.loginfo(f"start:\n{start}\ngoal:\n{goal}\ntolerance:\n{tolerance}\nexception:\n{e}")
            finally:
                # Gestisce il risultato della pianificazione
                if no_plan:
                    rospy.loginfo(f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} failed without plan.")
                    self.plan_check_retries += 1
                else:
                    rospy.loginfo(f"Plan check {self.plan_check_retries} out of {self.PLAN_CHECK_MAX_RETRIES} ok.")
                    self.plan_check_retries = 1
                self.last_plan_check = rospy.get_rostime().secs

    def handle_transition(self, gh):
        """Gestisce le transizioni di stato del goal"""
        # Verifica che il goal handle sia quello corretto
        if gh != self.gh:
            rospy.logerr("Got a transition callback on a goal handle that we're not tracking")
            return

        comm_state = gh.get_comm_state()
        rospy.loginfo(f"Transitioning with comm_state '{CommState.to_string(comm_state)}', simple_state '{SimpleGoalState.to_string(self.simple_state)}'")

        error_msg = "Received comm state %s when in simple state %s with SimpleActionClient in NS %s" % \
            (CommState.to_string(comm_state), SimpleGoalState.to_string(self.simple_state), rospy.resolve_name(self.action_client.ns))

        # Gestisce le varie transizioni di stato
        if comm_state == CommState.ACTIVE:
            if self.simple_state == SimpleGoalState.PENDING:
                self.simple_state = SimpleGoalState.ACTIVE
                if self.odom is not None:  # Verifica che odom sia disponibile
                    rospy.loginfo(f"Starting recovery behavior at position: x:{self.odom.position.x:.2f} y:{self.odom.position.y:.2f} Node: {self.goal_cnt}")
                else:
                    rospy.logwarn("Recovery behavior started but odometry data not available")
                self.active_cb()
            elif self.simple_state == SimpleGoalState.DONE:
                self.simple_state = SimpleGoalState.ACTIVE
                self.active_cb()
        elif comm_state == CommState.RECALLING:
            if self.simple_state != SimpleGoalState.PENDING:
                rospy.logerr(error_msg)
        elif comm_state == CommState.PREEMPTING:
            if self.simple_state == SimpleGoalState.PENDING:
                self.simple_state = SimpleGoalState.ACTIVE
                self.active_cb()
            elif self.simple_state == SimpleGoalState.DONE:
                rospy.logerr(error_msg)
        elif comm_state == CommState.DONE:
            if self.simple_state in [SimpleGoalState.PENDING, SimpleGoalState.ACTIVE]:
                self.done_cb(gh.get_goal_status(), gh.get_result())
                self.simple_state = SimpleGoalState.DONE
            elif self.simple_state == SimpleGoalState.DONE:
                rospy.logerr("SimpleActionClient received DONE twice")

    def done_cb(self, status, result):
        """Gestisce il completamento di un goal"""
        self.goal_cnt += 1

        '''
        Reference for terminal status values: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
        uint8 PENDING         = 0   # The goal has yet to be processed by the action server
        uint8 ACTIVE          = 1   # The goal is currently being processed by the action server
        uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing
                                    #   and has since completed its execution (Terminal State)
        uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)
        uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due
                                    #    to some failure (Terminal State)
        uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,
                                    #    because the goal was unattainable or invalid (Terminal State)
        uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing
                                    #    and has not yet completed execution
        uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,
                                    #    but the action server has not yet confirmed that the goal is canceled
        uint8 RECALLED        = 8   # The goal received a cancel request before it started executing
                                    #    and was successfully cancelled (Terminal State)
        uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be
                                    #    sent over the wire by an action server
        '''

        # Gestisce i vari stati di completamento
        if status == GoalStatus.PREEMPTED:
            if self.goal_cnt < len(self.pose_seq):
                rospy.loginfo(f"Goal pose {self.goal_cnt-1} received a cancel request after it started executing, completed execution.")
            else:
                rospy.loginfo("Final goal preempted!")
                rospy.signal_shutdown("Final goal preempted!")
                return
        elif status == GoalStatus.SUCCEEDED:
            rospy.loginfo(f"Goal pose {self.goal_cnt-1} reached.") 
            if self.goal_cnt < len(self.pose_seq):
                self.send_goal()
            else:
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return
        elif status == GoalStatus.ABORTED:
            rospy.loginfo(f"Goal pose {self.goal_cnt-1} was aborted by the Action Server. Skipping it")
            if self.goal_cnt < len(self.pose_seq):
                self.send_goal()
            else:
                rospy.loginfo("Final goal pose aborted.")
                rospy.signal_shutdown("Final goal pose aborted.")
            return
        elif status == GoalStatus.REJECTED:
            if self.goal_cnt < len(self.pose_seq):
                rospy.loginfo(f"Goal pose {self.goal_cnt-1} has been rejected by the Action Server")
                rospy.signal_shutdown(f"Goal pose {self.goal_cnt-1} rejected, shutting down!")
            else:
                rospy.loginfo("Final goal pose rejected.")
                rospy.signal_shutdown("Final goal pose rejected.")
            return
        elif status == GoalStatus.RECALLED:
            if self.goal_cnt < len(self.pose_seq):
                rospy.loginfo(f"Goal pose {self.goal_cnt-1} received a cancel request before it started executing, successfully cancelled!")
            else:
                rospy.loginfo("Final goal pose recalled.")
                rospy.signal_shutdown("Final goal pose recalled.")
                return

    def handle_feedback(self, gh, feedback):
        """Gestisce il feedback del goal"""
        if not self.gh:
            # Non è un errore - ci può essere una piccola finestra in cui viene ricevuto
            # feedback vecchio tra il reset della variabile e l'invio di un nuovo goal
            return
        if gh != self.gh:
            rospy.logerr("Got a feedback callback on a goal handle that we're not tracking. %s vs %s" %
                         (self.gh.comm_state_machine.action_goal.goal_id.id, gh.comm_state_machine.action_goal.goal_id.id))
            return
        self.feedback_cb(feedback)

    def get_current_goal(self):
        """Crea un nuovo goal per move_base"""
        goal = MoveBaseGoal()
        goal.target_pose.header.seq = self.goal_cnt
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now() 
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        return goal

    def send_goal(self):
        """Invia un nuovo goal a move_base"""
        goal = self.get_current_goal()
        rospy.loginfo(f"Sending goal pose {self.goal_cnt} to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))       
        self.gh = None
        self.simple_state = SimpleGoalState.PENDING
        self.gh = self.action_client.send_goal(goal, self.handle_transition, self.handle_feedback)
        self.goal_start = rospy.get_rostime().secs
        self.last_plan_check = self.goal_start
        self.plan_check_retries = 1

def parse_args():
    """Parsa gli argomenti da linea di comando"""
    parser = argparse.ArgumentParser(description='Send a list of waypoints to move_base.')
    parser.add_argument('-p', '--path', required=True, help="Path to the waypoints csv file.")
    return parser.parse_known_args()[0]

def read_waypoints(file_path):
    """Legge i waypoint da un file CSV"""
    waypoints = list()
    with open(file_path, "r") as file:
        pos = [l.strip().split(",") for l in file.readlines()]
        [waypoints.append(Point(float(x.strip()), float(y.strip()), 0)) for (x, y) in pos]
    return waypoints

def main():
    """Funzione principale"""
    # Parsa gli argomenti
    args = parse_args()
    csv_file_path = args.path

    # Legge i waypoint
    waypoints = read_waypoints(csv_file_path)
    
    # Inizializza il nodo ROS
    rospy.init_node('waypoint_sender', anonymous=True)
    _ = waypoint_sender(waypoints)

    # Mantiene il nodo attivo
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")

