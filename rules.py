# [<Radars: [46, 54, 293, 114, 73], Action: 1 ,Position: [830, 920]>, <Radars: [41, 52, 276, 114, 75], Action: 1 ,Position: [849.6961550602441, 923.4729635533386]>]
import math
import torch

NUMBER_OF_RULES = 2
SEGMENT_DISTRIBUTION_BY_RULES = [1 / 4, 1 / 4, 1 / 2]
PARTIAL_REWARD = False


def check_rules_flattened_one(segment, total_rules):
    rule_counter = 0

    def evaluate(rule_number, rule_lambda):
        increment = 1 if rule_lambda() else 0
        return increment if total_rules >= rule_number else 0

    # Rule 1: Check if the distance between two points is greater than 30
    point1 = segment[6:8]
    point2 = segment[14:16]
    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    rule_counter += evaluate(1, lambda: distance > 30)

    # Rule 2: Check if actions are not the same
    action1 = segment[5]
    action2 = segment[13]
    rule_counter += evaluate(2, lambda: action1 != action2)

    # Rule 3: Check if left radar is greater than right radar
    left_radar = segment[0]
    right_radar = segment[12]
    rule_counter += evaluate(3, lambda: left_radar > right_radar)

    return rule_counter
        


def check_rules_one(segment, total_rules):
    rule_counter = 0
    rules_followed = []

    def evaluate(rule_number, rule_lambda):
        increment = 1 if rule_lambda() else 0
        if rule_lambda() and total_rules >= rule_number:
            rules_followed.append(rule_number)
        return increment if total_rules >= rule_number else 0

    # Rule 1: Check if the distance between two points is greater than 30
    point1 = segment[0].position
    point2 = segment[1].position
    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    rule_counter += evaluate(1, lambda: distance > 30)

    # Rule 2: Check if actions are not the same
    action1 = segment[0].action
    action2 = segment[1].action
    rule_counter += evaluate(2, lambda: action1 != action2)

    # Rule 3: Check if left radar is greater than right radar
    left_radar = segment[1].radars[0]
    right_radar = segment[1].radars[4]
    rule_counter += evaluate(3, lambda: left_radar > right_radar)

    return (
        rule_counter,
        rule_counter if PARTIAL_REWARD else int(rule_counter == total_rules),
        rules_followed if rules_followed != [] else [0],
    )


def check_rules_long_segment(segment, total_rules):
    rule_counts = []
    total_reward = 0
    rules_followed_list = []
    for i in range(len(segment) - 1):
        rule_counter, reward, rules_followed = check_rules_one(
            segment[i : i + 2], total_rules
        )
        rule_counts.append(rule_counter)
        total_reward += reward
        rules_followed_list.append(rules_followed)
    return rule_counts, total_reward, rules_followed_list

def check_batch_rules(batch_segments, total_rules):
    batch_rule_counts = []
    
    for segment in batch_segments: 
        segment = segment.squeeze()
        rule_counter = check_rules_flattened_one(segment, total_rules)
        batch_rule_counts.append(rule_counter)

    return batch_rule_counts