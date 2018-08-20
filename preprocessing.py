from soynlp.word import WordExtractor as we


word_extractor = we(min_count=100,
                    min_cohesion_forward=.05,
                    min_right_branching_entropy=.0)

