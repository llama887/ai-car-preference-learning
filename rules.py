# [<Radars: [46, 54, 293, 114, 73], Action: 1 ,Position: [830, 920]>, <Radars: [41, 52, 276, 114, 75], Action: 1 ,Position: [849.6961550602441, 923.4729635533386]>]
import math

def check_rules(segment, total_rules):
    rule_counter = 0
    point1 = segment[0].position
    point2 = segment[1].position
    distance = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    if total_rules >= 1 and distance > 30:
        rule_counter += 1

    left_radar = segment[1].radars[0]
    right_radar = segment[1].radars[4]
    if total_rules >= 2 and left_radar > right_radar:
        rule_counter += 1
    
    return rule_counter, rule_counter == total_rules