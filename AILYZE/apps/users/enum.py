from enum import Enum

class Anaylsis(Enum):
    Summarize = "Summarize"
    Ask_a_specific_question = "Ask_a_specific_question"
    Conduct_thematic_analysis = "Conduct_thematic_analysis"
    Identidy_which_document_contain_a_certain_viewpoint = "Identidy_which_document_contain_a_certain_viewpoint"
    Compare_viewpoints_across_documents="Compare_viewpoints_across_documents"


    @classmethod
    def choices(cls):
        return {i.name:i.value for i in Anaylsis}


class Excelchoice(Enum):
    Conduct_thematic_analysis_based_on_text_in_a_column = "Conduct_thematic_analysis_based_on_text_in_a_column"
    Categorize_text_in_each_cell_in_a_column = "Categorize_text_in_each_cell_in_a_column"

    @classmethod
    def choices(cls):
        return {i.name:i.value for i in Excelchoice}

