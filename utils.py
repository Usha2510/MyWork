from typing import List
import re

import ast

greetings_list = [
    "hi",
    "hello",
    "hey",
    "helloo",
    "welcome",
    "hellooo",
    "g morining",
    "gmorning",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "greeting",
    "good to see you",
    "its good seeing you",
    "how is your day",
    "how are you",
    "how're you",
    "how are you doing",
    "how ya doin'",
    "how ya doin",
    "how is everything",
    "how is everything going",
    "how's everything going",
    "how is you",
    "how's you",
    "how are things",
    "how're things",
    "how is it going",
    "how's it going",
    "how's it goin'",
    "how's it goin",
    "how is life been treating you",
    "how's life been treating you",
    "how have you been",
    "how've you been",
    "what is up",
    "what's up",
    "what is cracking",
    "what's cracking",
    "what is good",
    "what's good",
    "what is happening",
    "what's happening",
    "what is new",
    "what's new",
    "what is neww",
    "g’day",
    "howdy",
    "what can I do for you",
    "How can I help you",
]
ADD_WELCOME = ["you are welcome", "you're welcome", "You're welcome"]
bye_list = [
    "bye",
    "see you later",
    "bye",
    "see you later",
    "goodbye",
    "nice chatting to you, bye",
    "have a nice day",
    "see you soon",
    "good-bye",
    "see you tomorrow",
    "have a wonderful day" "have a good day",
]

keywords_list = [
    "wash as often as you like",
    "unlimited car washes",
    "cancel any time",
    "skip the line",
    "no contract",
    "feel great driving a clean car",
    "cancel anytime",
    "unlimited wash",
    "unlimited car wash",
    "unlimited",
    "unlimit",
    "wash",
    "often",
    "skip",
    "contract",
    "clean",
    "cancel",
    "limited",
    "membership",
    "special",
    "barcode",
    "sticker",
]


def create_pattern(words_list: List) -> str:
    joined_list = "|".join(words_list).replace("$", "")
    return rf"\b({joined_list})\b".lower().replace("$", "\$")


pattern_greeting = create_pattern(greetings_list)
pattern_bye = create_pattern(bye_list)
pattern_engagement = create_pattern(keywords_list)


def split_conversation(utterance, prev_transcript_list):
    """ Finds if any multiple conversation possible in same utterance """
    match_greeting_list = re.findall(pattern_greeting, utterance, re.IGNORECASE)
    match_bye_list = re.findall(pattern_bye, utterance, re.IGNORECASE)
    conversation_list = []

    if len(match_greeting_list) == 1:
        return conversation_list

    # Scenario1: Check if the first greeting starts at 0th pos
    previous_conv_remain = False
    previous_utterance = " -- ".join(prev_transcript_list)
    if (
        len(prev_transcript_list) > 1 and utterance.index(match_greeting_list[0]) != 0
    ):  # utterance[utterance.index(match_greeting_list[0])] != 0:
        # Scenario2: if bye/end is before the greet
        if len(match_bye_list) > 0 and utterance.index(
            match_greeting_list[0]
        ) > utterance.index(match_bye_list[0]):
            previous_utterance += (
                " -- "
                + utterance[0 : utterance.index(match_bye_list[0])]
                + match_bye_list[0]
            )
            conversation_list.append(previous_utterance)
        # Scenario ended
        else:
            previous_utterance += (
                " -- " + utterance[0 : utterance.index(match_greeting_list[0])]
            )
            conversation_list.append(previous_utterance)
        previous_conv_remain = True
    # Scenario ended

    # Set the next position to start the conversation
    start_greet_pos = (
        utterance.index(match_greeting_list[0]) if previous_conv_remain else 0
    )
    for pattern in match_greeting_list[1:]:
        split_conv = utterance[start_greet_pos : utterance.index(pattern)]
        utterance = utterance[utterance.index(pattern) :]
        conversation_list.append(split_conv)
    split_conv = utterance[0:]
    conversation_list.append(split_conv)

    # If the length of the split conv is less than 10 then combine with next split conv
    i = 0
    while i < len(conversation_list):
        split_conv = conversation_list[i]
        if len(split_conv.replace(" -- ", " ").split(" ")) < 10:
            if i + 1 < len(conversation_list):
                conversation_list[i + 1] = (
                    conversation_list[i] + conversation_list[i + 1]
                )
                if i == 0:
                    # left shift the list to 0-th position by skipping 0-th element
                    conversation_list = conversation_list[i + 1 :]
                else:
                    # Skip the i-th element
                    if i - 1 == 0:
                        conversation_list = [conversation_list[0]] + conversation_list[
                            i + 1 :
                        ]
                    else:
                        conversation_list = (
                            conversation_list[0 : i - 1] + conversation_list[i + 1 :]
                        )
                i = 0
                continue
        i += 1

    return conversation_list


def get_engagements(utterance_data, next_engg_id, conversation_id) -> List:
    match_flag = 0
    engagements = []
    s = []
    ai_master_utterance_data = []
    for utterance in utterance_data:
        transcript = utterance[0]
        started_at = utterance[1]
        ended_at = utterance[2]
        s3_key = utterance[3]
        transcript = transcript.replace('"', "")
        match_greeting = re.search(pattern_greeting, transcript, re.IGNORECASE)
        match_bye = re.search(pattern_bye, transcript, re.IGNORECASE)
        match_engagement = re.search(pattern_engagement, transcript, re.IGNORECASE)
        NULL_DATE = "NULL"

        if match_greeting:
            # : Special handling for mulit conversation in same utterance
            split_conv_list = split_conversation(transcript, s)
            if len(split_conv_list) > 1:
                # Insert all except last to the conversation table list
                # Insert the split utterance info to the master AI table list
                # Get the match_bye for each conversation
                for split_conv in split_conv_list[:-1]:
                    transcript = split_conv
                    match_bye = re.search(pattern_bye, transcript, re.IGNORECASE)

                    ai_master_utterance_data.append(
                        [
                            next_engg_id,
                            conversation_id,
                            started_at,
                            ended_at,
                            split_conv,
                            s3_key,
                        ]
                    )
                    if match_bye:
                        engagements.append(
                            [
                                next_engg_id,
                                conversation_id,
                                split_conv,
                                NULL_DATE,
                                NULL_DATE,
                                1,
                            ]
                        )
                        next_engg_id += 1
                    else:
                        engagements.append(
                            [
                                next_engg_id,
                                conversation_id,
                                split_conv,
                                NULL_DATE,
                                NULL_DATE,
                                0,
                            ]
                        )
                    conversation_id += 1
                # Reset the previous utterance list
                s = []

                # Keep the current transript as the last from the list
                transcript = split_conv_list[-1]
                # Get the match_bye and match_engagement for the transcript again
                match_bye = re.search(pattern_bye, transcript, re.IGNORECASE)
                match_engagement = re.search(
                    pattern_engagement, transcript, re.IGNORECASE
                )
                # Continue with same logic treating last as the next utterance
            # END

            match_flag = 1
            # To not confuse this with a greeting "welcome"
            if not re.search(
                rf"\b(you are welcome|you're welcome|You're welcome)\b",
                transcript,
                re.IGNORECASE,
            ):
                conversation = " -- ".join(s)
                if len(s) > 0:
                    if (
                        len(conversation.replace(" -- ", " ").split(" ")) > 10
                    ):  # and re.search(pattern_engagement, conversation, re.IGNORECASE)):
                        new_match_bye = re.search(
                            pattern_bye, conversation, re.IGNORECASE
                        )
                        if new_match_bye:
                            engagements.append(
                                [
                                    next_engg_id,
                                    conversation_id,
                                    conversation,
                                    NULL_DATE,
                                    NULL_DATE,
                                    1,
                                ]
                            )
                        else:
                            engagements.append(
                                [
                                    next_engg_id,
                                    conversation_id,
                                    conversation,
                                    NULL_DATE,
                                    NULL_DATE,
                                    0,
                                ]
                            )

                        conversation_id += 1
                s = []

        if match_engagement:
            match_flag = 1

        if match_bye:
            if s:
                s.append(transcript)
                ai_master_utterance_data.append(
                    [
                        next_engg_id,
                        conversation_id,
                        started_at,
                        ended_at,
                        transcript,
                        s3_key,
                    ]
                )
                conversation = " -- ".join(s)
                if (
                    len(conversation.replace(" -- ", " ").split(" ")) > 10
                ):  # and re.search(pattern_engagement, conversation, re.IGNORECASE)):
                    engagements.append(
                        [
                            next_engg_id,
                            conversation_id,
                            conversation,
                            NULL_DATE,
                            NULL_DATE,
                            1,
                        ]
                    )
                    conversation_id += 1
                next_engg_id += 1
                s = []
                match_flag = 0

        if match_flag:
            s.append(transcript)
            ai_master_utterance_data.append(
                [
                    next_engg_id,
                    conversation_id,
                    started_at,
                    ended_at,
                    transcript,
                    s3_key,
                ]
            )

    if s:
        conversation = " -- ".join(s)
        if (
            len(conversation.replace(" -- ", " ").split(" ")) > 10
        ):  # and re.search(pattern_engagement, conversation, re.IGNORECASE)):
            new_match_bye = re.search(pattern_bye, conversation, re.IGNORECASE)
            if new_match_bye:
                engagements.append(
                    [
                        next_engg_id,
                        conversation_id,
                        conversation,
                        NULL_DATE,
                        NULL_DATE,
                        1,
                    ]
                )
            else:
                engagements.append(
                    [
                        next_engg_id,
                        conversation_id,
                        conversation,
                        NULL_DATE,
                        NULL_DATE,
                        0,
                    ]
                )
            conversation_id += 1
        s = []
        match_flag = 0
    return engagements, ai_master_utterance_data


def find_superset(matches, records):
    superset = []
    for match in matches:
        for i, record in enumerate(records):
            for phrase in ast.literal_eval(record["phrases"]):
                phrase_without_dollar_sign = phrase.replace("$", "")
                if re.findall(match, phrase_without_dollar_sign, re.IGNORECASE):
                    superset.append((phrase_without_dollar_sign, i))
    superset = sorted(superset, key=lambda x: len(x[0]), reverse=True)
    return superset


def evaluate_engagements(engagements, records):
    # {"scores":[{"name":"Special Service","said":2,"weight":90,"required":2},{"name":"Welcome","said":1,"weight":10,"required":1}]}
    score_data = []
    score_info = {}
    for engagement in engagements:
        e = engagement[2].replace("$", "")  # Only the concatenated conversation is used
        weight = 0
        score_info = {}
        eng_score_info = []
        for record in records:
            eng_score = {
                "name": record["name"],
                "said": 0,
                "weight": 0,
                "required": record["min_phrases"],
            }
            matches = re.findall(record["pattern"], e, re.IGNORECASE)
            if matches:
                superset = find_superset(matches, records)
                for super in superset:
                    s_index = super[1]
                    s_matches = re.findall(
                        records[s_index]["pattern"], e, re.IGNORECASE
                    )
                    if len(s_matches) >= int(records[s_index]["min_phrases"]):

                        for s_match in s_matches:
                            e = e.replace(s_match, "").strip()

                        weight += records[s_index]["weight"]
                        eng_score["weight"] = records[s_index]["weight"]
                        eng_score["said"] = min(len(s_matches), eng_score["required"])
            eng_score_info.append(eng_score)
        score_info["scores"] = eng_score_info
        score_data.append([weight, score_info])

    return score_data