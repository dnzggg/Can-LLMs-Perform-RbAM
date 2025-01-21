import csv
import os


class RbAMPrompts:
    @staticmethod
    def deniz(arg1, arg2, nr=False, support_label="Support", attack_label="Attack", no_label="No", n_shot="0", primer=None, instruction=True, **_):
        def formatter(relation):
            logits = None
            if type(relation) == tuple:
                logits = relation[1]
                relation = relation[0]

            if (rel := relation.replace("Relation:", "").strip()) not in [support_label, attack_label, no_label]:
                if logits:
                    logits = {token.token: token.logprob for token in logits}
                    support=attack=no=-float('inf')
                    if support_label in logits.keys():
                        support = logits[support_label]
                    
                    if attack_label in logits.keys():
                        attack = logits[attack_label]

                    if no_label in logits.keys():
                        support = logits[no_label]

                    if support > attack and support > no:
                        return 1
                    elif attack > support and attack > no:
                        return 0
                    elif no > attack and no > support:
                        return 2
                return -1

            return (1 if rel == support_label else 0) if rel != no_label else 2

        constraints = {
            "constraint_prefix": "Relation:",
            "constraint_options": [support_label, attack_label] + ([no_label] if nr else []),
            "constraint_end_after_options": True,
        }

        instructions = (f"In this task, you will be given two arguments and your goal is to classify " +
                       (f"the relation between them as either “{support_label}”, or “{attack_label}” based on the " if not nr else
                        f"the relation between them as either “{support_label}”, “{attack_label}”, or “{no_label}” based on the ") +
                        f"definitions below.\n'{support_label}': It is an argument that is in favour of to the parent "
                        f"argument.\n'{attack_label}': It is an argument that contradicts or opposes the parent "
                        f"argument.\n" + (f"\n" if not nr else f"'{no_label}': It is an argument that has no relation "
                                                               f"to the parent argument.\n"))
        if not instruction:
            instructions = ""

        if n_shot != "0":
            file = open(os.path.dirname(os.path.abspath(__file__)) + f"/few_shot/{n_shot}/{primer}")
            reader = csv.reader(file, delimiter='#', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            primer = ""
            for row in reader:
                primer += (f"Arg1: {row[0]}\nArg2: {row[1]}\nRelation: "
                           f"{(support_label if row[2] == '1' else attack_label) if row[2] != '2' else no_label}\n\n")
        else:
            primer = ""

        prompt = instructions + primer + f"Arg1: {arg1}\nArg2: {arg2}"

        return prompt, constraints, formatter

    @staticmethod
    def chat_gpt(arg1, arg2, nr=False, support_label="Support", attack_label="Attack", no_label="No", n_shot=0, instruction=True, **_):
        def formatter(relation):
            if (rel := relation.replace("Relation:", "").strip()) not in [support_label, attack_label, no_label]:
                return -1
            return (1 if rel == support_label else 0) if rel != no_label else 2

        constraints = {
            "constraint_prefix": "Relation: ",
            "constraint_options": [support_label, attack_label] + (["No"] if nr else []),
            "constraint_end_after_options": True,
        }

        instructions = ""
        if instruction:
            instructions = (f"Imagine a contentious issue has ignited a fervent discussion, with individuals advocating "
                            f"diverse perspectives. Each side articulates arguments to bolster their position. "
                            f"Your role is to scrutinize a pair of texts, denoted as A and B, and ascertain the nature "
                            f"of their relationship. Does text A: '{arg1}' undermine the viewpoint expounded in text B: "
                            f"'{arg2}', " + (f"or does it endorse it?\n\n" if not nr else f"endorse it, or are they "
                                                                                          f"unrelated?\n\n"))

        # TODO: Randomly select examples from datasets, control number of few_shot
        primer = ""
        if n_shot != 0:
            primer = (f"Example:\n\tText A: 'Climate change is primarily caused by human activities such as burning "
                      f"fossil fuels and deforestation.'\n"
                      f"\tText B: 'Some argue that fluctuations in Earth's climate "
                      f"are part of natural cycles and not significantly impacted by human actions.'\n"
                      f"\tRelation: {support_label}\n\n")

        prompt = (instructions + primer + (f"Provide your analysis in the following format:\n\n"
                                           f"Relation: {support_label} if you believe text A endorses the viewpoint "
                                           f"in text B, " + (f"or " if not nr else f"") +
                                           f"{attack_label} if you think text A undermines the viewpoint in text B" +
                                           (f"" if not nr else f", or {no_label} if there's no clear connection "
                                                               f"between the two texts") +
                                           f".\n\nOffer your assessment grounded in a thorough examination of "
                                           f"the textual content and the coherence of their arguments.\n\n"
                                           f"Text A: {arg1}\nText B: {arg2}\n"))

        return prompt, constraints, formatter
