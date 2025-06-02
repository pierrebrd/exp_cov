"""
Script per confrontare navigazione basata su esplorazione vs copertura usando ROS e Stage.
Esegue esperimenti multipli confrontando due strategie:
1. Esplorazione usando explore_lite
2. Navigazione a copertura usando waypoint predefiniti

Per ogni test:
- Crea una directory per i dati
- Esegue esplorazione e misura tempo/area coperta
- Esegue copertura e misura tempo/area coperta
- Salva mappe e log per entrambe le strategie
"""

# Import necessari
import subprocess as sp
import argparse
import cv2
from PIL import Image
import numpy as np
import os
import rospy
from time import gmtime, strftime, sleep

"""
Analizza e valida argomenti linea comando.
@return: Oggetto con argomenti parsati contenente:
         - waypoints: Percorso file CSV waypoint
         - world: Percorso file mondo Stage
         - runs: Numero di test da eseguire
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Start exploration, logging time info to file and saving the final map.')
    parser.add_argument('--waypoints', required=True, help="Path to the waypoints csv file.")
    parser.add_argument('--world', required=True, help="Path to the stage world file.")
    parser.add_argument('-r', '--runs', required=False, default=1,  type=check_positive, help="Number of tests to run.", metavar="RUNS")
    parser.add_argument('-d', '--dir', required=False, default="", help="Directory to save the run data.", metavar="DIR")
    return parser.parse_args()

"""
Ottiene timestamp corrente in formato leggibile.
@return: Ora corrente in formato YYYY-MM-DD HH:MM:SS
"""
def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

"""
Esegue esplorazione usando explore_lite e salva la mappa risultante.
@param logfile_path: Percorso dove salvare i log
@param run_subfolder: Sottocartella per i dati del test corrente
@return: Tempo impiegato per l'esplorazione in secondi
"""
def run_expl(logfile_path, run_subfolder = ""):
    start = None
    args = ["roslaunch", "exp_cov", "explore_lite2.launch"]
    error_log_path = os.path.join(run_subfolder, "exploreErrorLog.txt")
    info_log_path = os.path.join(run_subfolder, "info.log")
    last_message = None
    last_message_time = 0
    
    with sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT) as process:
        with open(logfile_path, mode="+a", encoding="utf-8") as logfile, \
             open(error_log_path, mode="+a", encoding="utf-8") as error_log, \
             open(info_log_path, mode="+a", encoding="utf-8") as info_log:
            try:
                start = rospy.get_rostime().secs
                logfile.write(f"{now()}: Starting exploration.\n")
                for line in process.stdout:
                    line = line.decode('utf8')
                    current_time = rospy.Time.now().secs
                    
                    # Log tutte le informazioni non di errore
                    if not any(x in line.lower() for x in ["error", "abort", "stuck", "timeout"]):
                        # Controlla se il messaggio deve essere loggato
                        if should_log_message(line):
                            # Se il messaggio è diverso dal precedente o è passato più di 1 secondo
                            if line.strip() != last_message or (current_time - last_message_time) >= 1:
                                info_log.write(f"{now()}: {line.strip()}\n")
                                last_message = line.strip()
                                last_message_time = current_time
                    
                    # Check for error conditions in the output
                    if "error" in line.lower():
                        error_log.write(f"{now()}: Exploration Error - {line.strip()}\n")
                    if "abort" in line.lower():
                        error_log.write(f"{now()}: Exploration Aborted - {line.strip()}\n")
                    if "stuck" in line.lower():
                        error_log.write(f"{now()}: Robot Stuck - {line.strip()}\n")
                    if "timeout" in line.lower():
                        error_log.write(f"{now()}: Operation Timeout - {line.strip()}\n")
                        
                    if line.strip()[1:].startswith("["):
                        if "exploration stopped." in line.lower():
                            logfile.write(f"{now()}: Finished exploration.\n")
                            break
            except KeyboardInterrupt as e:
                error_msg = f"{now()}: Exploration Interrupted by user.\n"
                logfile.write(error_msg)
                error_log.write(error_msg)
            except Exception as e:
                error_msg = f"{now()}: Unexpected error during exploration: {str(e)}\n"
                error_log.write(error_msg)
            finally:
                time = rospy.get_rostime().secs - start
                logfile.write(f"{now()}: Exploration ros time is {strftime('%H:%M:%S', gmtime(time))}.\n")
                process.kill()
                map_name = os.path.join(run_subfolder, "Map_exploration")
                save_map = ["rosrun", "map_server", "map_saver", "-f", map_name]
                sp.run(save_map)
                try:
                    Image.open(f"{map_name}.pgm").save(f"{map_name}.png")
                    os.remove(f"{map_name}.pgm")
                except IOError:
                    print("Cannot convert pgm map to png.")
                finally:
                    return time

"""
Determina se un messaggio deve essere loggato.
@param line: Linea di log da valutare
@return: True se il messaggio va loggato, False altrimenti
"""
def should_log_message(line):
    # Lista di messaggi significativi da loggare
    important_messages = [
        "waypoint sender started",
        "connected to move_base server",
        "sending goal pose",
        "goal pose reached",
        "found frontiers",
        "visualising frontiers",
        "waiting for costmap"
    ]
    
    # Ignora completamente certi tipi di messaggi
    ignore_messages = [
        "tf_repeated_data",
        "getting status over the wire",
        "debug",
        "trying to publish",
        "transitioning",
        "received comm state"
    ]
    
    line_lower = line.lower()
    
    # Se il messaggio contiene una delle stringhe da ignorare, non loggarlo
    if any(x in line_lower for x in ignore_messages):
        return False
        
    # Se il messaggio contiene una delle stringhe importanti, loggarlo
    return any(x in line_lower for x in important_messages)

"""
Esegue navigazione a waypoint e salva la mappa risultante.
@param waypoints: Percorso file CSV waypoint
@param logfile_path: Percorso dove salvare i log
@param run_subfolder: Sottocartella per i dati del test corrente
@return: Tempo impiegato per la navigazione in secondi
"""
def run_cov(waypoints, logfile_path="./coverage.log", run_subfolder = ""):
    start = None
    args = ["rosrun", "exp_cov", "waypoint_navigation.py", "-p", waypoints]
    error_log_path = os.path.join(run_subfolder, "coverageErrorLog.txt")
    info_log_path = os.path.join(run_subfolder, "info.log")
    last_message = None
    last_message_time = 0
    
    with sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT) as process:
        with open(logfile_path, mode="+a", encoding="utf-8") as logfile, \
             open(error_log_path, mode="+a", encoding="utf-8") as error_log, \
             open(info_log_path, mode="+a", encoding="utf-8") as info_log:
            try:
                start = rospy.get_rostime().secs
                logfile.write(f"{now()}: Starting waypoint navigation.\n")
                for line in process.stdout:
                    line = line.decode('utf8')
                    current_time = rospy.Time.now().secs
                    
                    # Log tutte le informazioni non di errore
                    if not any(x in line.lower() for x in ["error", "abort", "timeout", "recovery"]):
                        # Controlla se il messaggio deve essere loggato
                        if should_log_message(line):
                            # Se il messaggio è diverso dal precedente o è passato più di 1 secondo
                            if line.strip() != last_message or (current_time - last_message_time) >= 1:
                                info_log.write(f"{now()}: {line.strip()}\n")
                                last_message = line.strip()
                                last_message_time = current_time

                    # Check for error conditions
                    if "error" in line.lower():
                        error_log.write(f"{now()}: Navigation Error - {line.strip()}\n")
                    if "abort" in line.lower():
                        error_log.write(f"{now()}: Navigation Aborted - {line.strip()}\n")
                    if "timeout" in line.lower():
                        error_log.write(f"{now()}: Operation Timeout - {line.strip()}\n")
                    if "recovery" in line.lower():
                        error_log.write(f"{now()}: Recovery Action - {line.strip()}\n")
                        print("starting recovery behavior")

                    if "final goal" in line.lower():
                        logfile.write(f"{now()}: Finished waypoint navigation.\n")
                        break
            except KeyboardInterrupt as e:
                error_msg = f"{now()}: Waypoint navigation Interrupted by user.\n"
                logfile.write(error_msg)
                error_log.write(error_msg)
            except Exception as e:
                error_msg = f"{now()}: Unexpected error during navigation: {str(e)}\n"
                error_log.write(error_msg)
            finally:
                time = rospy.get_rostime().secs - start
                logfile.write(f"{now()}: Waypoint navigation ros time is {strftime('%H:%M:%S', gmtime(time))}.\n")
                process.kill()
                map_name = os.path.join(run_subfolder, "Map_coverage")
                save_map = ["rosrun", "map_server", "map_saver", "-f", map_name]
                sp.run(save_map)
                try:
                    Image.open(f"{map_name}.pgm").save(f"{map_name}.png")
                    os.remove(f"{map_name}.pgm")
                except IOError:
                    print("Cannot convert pgm map to png.")
                finally:
                    return time

"""
Configura ed esegue il processo di esplorazione completo.
Avvia i nodi ROS necessari (Stage, SLAM, distance checker)
ed esegue la strategia di esplorazione.
@param cmd_args: Argomenti linea comando
@param logfile_path: Percorso per logging
@param run_subfolder: Sottocartella per i dati del test
@return: Tempo totale esplorazione in secondi
"""
def run_exploration(cmd_args, logfile_path, run_subfolder):
    print("starting exploration.")
    stage_args = ["roslaunch", "exp_cov", "stage_init.launch", f"worldfile:={cmd_args.world}"]
    slam_args = ["roslaunch", "exp_cov", "slam_toolbox_no_rviz.launch"]
    dist_args = ["rosrun", "exp_cov", "distance_check.py"]
    with sp.Popen(stage_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as stage_process:
        sleep(3)
        print("started stage.")
        with sp.Popen(slam_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as slam_process:
            sleep(10)
            print("started slam.")
            with open(logfile_path, mode="+a", encoding="utf-8") as logfile:
                with sp.Popen(dist_args, stdout=logfile, stderr=logfile) as dist_process:
                    time = run_expl(logfile_path, run_subfolder)
                    print("exploration finished.")
                    dist_process.terminate()
                    slam_process.terminate()
                    stage_process.terminate()
                    return time

"""
Configura ed esegue il processo di copertura completo.
Avvia i nodi ROS necessari (Stage, SLAM, distance checker)
ed esegue la strategia di copertura a waypoint.
@param cmd_args: Argomenti linea comando
@param logfile_path: Percorso per logging
@param run_subfolder: Sottocartella per i dati del test
@return: Tempo totale copertura in secondi
"""
def run_coverage(cmd_args, logfile_path, run_subfolder):
    print("starting coverage.")
    stage_args = ["roslaunch", "exp_cov", "stage_init.launch", f"worldfile:={cmd_args.world}"]
    slam_args = ["roslaunch", "exp_cov", "waypoint_slam.launch"]
    dist_args = ["rosrun", "exp_cov", "distance_check.py"]
    with sp.Popen(stage_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as stage_process:
        sleep(3)
        print("started stage.")
        with sp.Popen(slam_args, stdout=sp.DEVNULL, stderr=sp.DEVNULL) as slam_process:
            sleep(10)
            print("started slam.")
            with open(logfile_path, mode="+a", encoding="utf-8") as logfile:
                with sp.Popen(dist_args, stdout=logfile, stderr=logfile) as dist_process:
                    time = run_cov(cmd_args.waypoints, logfile_path, run_subfolder)
                    print("coverage finished.")
                    dist_process.terminate()
                    slam_process.terminate()
                    stage_process.terminate()
                    return time

"""
Valida che un valore sia un intero positivo.
@param value: Valore da controllare
@return: Intero positivo validato
@raises: Exception se valore non valido
"""
def check_positive(value):
    """
    Validate that a string represents a positive integer.

    Args:
        value: Value to check

    Returns:
        int: The validated positive integer

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer
        Exception: If value cannot be converted to integer
    """
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("{} is not a positive integer".format(value))
    except ValueError:
        raise Exception("{} is not an integer".format(value))
    return value

"""
Logica principale per il confronto esplorazione vs copertura.
Per ogni test:
1. Crea directory per i dati
2. Esegue test di esplorazione e copertura
3. Confronta e registra:
   - Differenze tempo tra strategie
   - Differenze copertura area
   - Salva mappe risultanti
@param cmd_args: Argomenti parsati con parametri test
"""
def main(cmd_args):
    logfile_path_exploration = "explore.log"
    logfile_path_coverage = "coverage.log"
    logfile_path_result = "result.log"

    # Crea la directory parent se specificata e non esiste
    parent_dir = cmd_args.dir if cmd_args.dir else "."
    if parent_dir != "." and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Trova il numero massimo di run nella directory parent
    maxrun = 0
    for i in os.listdir(parent_dir):
        try:
            if int(i[3:]) >= maxrun:
                maxrun = int(i[3:]) + 1
        except:
            continue

    time_deltas = list()
    area_deltas = list()
    for r in range(int(cmd_args.runs)):
        print(f"run {r+1}/{cmd_args.runs} starting.")
        run_subfolder = os.path.join(parent_dir, f"run{maxrun+r}")
        os.mkdir(run_subfolder)
        logfile_path_exploration_run = os.path.join(run_subfolder, logfile_path_exploration)
        logfile_path_coverage_run = os.path.join(run_subfolder, logfile_path_coverage)
        logfile_path_result_run = os.path.join(run_subfolder, logfile_path_result)
        exploration_time = run_exploration(cmd_args, logfile_path_exploration_run, run_subfolder)
        sleep(2)
        coverage_time = run_coverage(cmd_args, logfile_path_coverage_run, run_subfolder)
        with open(logfile_path_result_run, mode="+a", encoding="utf-8") as logfile:
            time_delta = exploration_time-coverage_time
            time_deltas.append(exploration_time-coverage_time)
            msg = f"{now()}: Coverage time: {coverage_time}; Exploration time: {exploration_time}. Exploration - Coverage: {time_delta}. Unit is seconds."
            print(msg)
            logfile.write(f"{msg}\n")
            expl_map = cv2.imread(os.path.join(run_subfolder, "Map_exploration.png"), cv2.IMREAD_GRAYSCALE)
            cov_map = cv2.imread(os.path.join(run_subfolder, "Map_coverage.png"), cv2.IMREAD_GRAYSCALE)
            expl_map_area = np.sum(expl_map >= 250)
            cov_map_area = np.sum(cov_map >= 250)
            area_delta = expl_map_area-cov_map_area
            area_deltas.append(area_delta)
            msg = f"{now()}: Coverage mapped area: {cov_map_area}; Exploration mapped area: {expl_map_area}. Exploration - Coverage: {area_delta}. Unit is 0.05 meters, a pixel in the map."
            print(msg)
            logfile.write(f"{msg}\n")
        print(f"run {r+1}/{cmd_args.runs} finished.")
    print(f"time_deltas (exploration-coverage): {time_deltas}\n;\narea_deltas (exploration-coverage): {area_deltas}\n;\n")

if __name__ == "__main__":

    cmd_args = parse_args()
    with sp.Popen(["roscore"], stdout=sp.DEVNULL, stderr=sp.DEVNULL) as roscore_process:
        sleep(3)
        rospy.set_param('use_sim_time', True)
        rospy.init_node('just_for_time', anonymous=True)
        sleep(3)
        main(cmd_args)
        roscore_process.kill()
