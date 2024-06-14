from dataclasses import dataclass, field
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader

from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

from ebrec.utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)


@dataclass
class NewsrecDataLoader(Dataset):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)






@dataclass(kw_only=True)
class PPRecDataLoader(NewsrecDataLoader):
    # unknown_category_value: int = 0
    # unknown_subcategory_value: int = 0
    body_mapping: dict[int, list[int]] = None
    popularity_mapping: dict[int, int]= None
    # category_mapping: dict[int, int] = None
    # subcategory_mapping: dict[int, int] = None

    def __post_init__(self):
        self.title_prefix = "title_"
        self.body_prefix = "ner_clusters_text_"
        self.popularity_prefix = 'views_'
        # self.category_prefix = "category_"
        # self.subcategory_prefix = "subcategory_"
        # (
        #     self.lookup_article_index_body,
        #     self.lookup_article_matrix_body,
        # ) = create_lookup_objects(
        #     self.body_mapping, unknown_representation=self.unknown_representation
        # )
        # if self.eval_mode:
        #     raise ValueError("'eval_mode = True' is not implemented for NAML")
        

        return super().__post_init__()

    def transform(self, df: pl.DataFrame) -> tuple[pl.DataFrame]:
        """
        Special case for NAML as it requires body-encoding, verticals, & subvertivals
        """
        # =>
        title = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        # =>
        # body = df.pipe(
        #     map_list_article_id_to_value,
        #     behaviors_column=self.history_column,
        #     mapping=self.lookup_article_index_body,
        #     fill_nulls=self.unknown_index,
        #     drop_nulls=False,
        # ).pipe(
        #     map_list_article_id_to_value,
        #     behaviors_column=self.inview_col,
        #     mapping=self.lookup_article_index_body,
        #     fill_nulls=self.unknown_index,
        #     drop_nulls=False,
        # )
        # popularity = df.pipe(
        #     map_list_article_id_to_value,
        #     behaviors_column=self.history_column,
        #     mapping=self.popularity_mapping,
        #     fill_nulls=0,
        #     drop_nulls=False,
        # ).pipe(
        #     map_list_article_id_to_value,
        #     behaviors_column=self.inview_col,
        #     mapping=self.popularity_mapping,
        #     fill_nulls=0,
        #     drop_nulls=False,
        # )
        
      
        return (
            pl.DataFrame()
            .with_columns(title.select(pl.all().name.prefix(self.title_prefix)))
            # .with_columns(body.select(pl.all().name.prefix(self.body_prefix)))
            # .with_columns(popularity.select(pl.all().name.prefix(self.popularity_prefix)))
            
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            #print(batch_X.columns)
            repeats_title = np.array(batch_X["title_n_samples"])
            # repeats_body = np.array(batch_X["ner_clusters_text_n_samples"])
            # repeats_pop = np.array(batch_X["views_n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.title_prefix + self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats_title,
            )
            
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.title_prefix + self.inview_col].explode().to_list()
            ]

            # his_input_body = repeat_by_list_values_from_matrix(
            #     batch_X[self.body_prefix + self.history_column].to_list(),
            #     matrix=self.lookup_article_matrix,
            #     repeats=repeats_body,
            # )
            # # =>
            # pred_input_body = self.lookup_article_matrix[
            #     batch_X[self.body_prefix + self.inview_col].explode().to_list()
            # ]

            # his_input_pop = repeat_by_list_values_from_matrix(
            #     batch_X[self.popularity_prefix + self.history_column].to_list(),
            #     matrix=self.lookup_article_matrix,
            #     repeats=repeats_pop,
            # )
            # # =>
            # pred_input_pop = self.lookup_article_matrix[
            #     batch_X[self.popularity_prefix + self.inview_col].explode().to_list()
            # ]
           
            his_input_title = np.squeeze(
                his_input_title, axis=2
                )
                
            # his_input_body = np.squeeze(
            #         his_input_body, axis=2
            #         )
            # his_input_pop = np.squeeze(
            #         his_input_pop, axis=2
            #         )
            
            

        else:
            batch_y = np.array(batch_y.to_list())
            
            his_input_title = np.array(
                batch_X[self.title_prefix + self.history_column].to_list()
            )
            # his_input_body = np.array(
            #     batch_X[self.body_prefix + self.history_column].to_list()
            # )
            # his_input_pop = np.array(
            #     batch_X[self.popularity_prefix + self.history_column].to_list()
            # )
            
            pred_input_title = np.array(
                batch_X[self.title_prefix + self.inview_col].to_list()
            )
            # pred_input_body = np.array(
            #     batch_X[self.body_prefix + self.inview_col].to_list()
            # )
            # pred_input_pop = np.array(
            #     batch_X[self.popularity_prefix + self.inview_col].to_list()
            # )
            
            
            pred_input_title = np.squeeze(
                self.lookup_article_matrix[pred_input_title], axis=2
            )
            
            # pred_input_body = np.squeeze(
            #     self.lookup_article_matrix_body[pred_input_body], axis=2
            # )
            # pred_input_pop = np.squeeze(
            #     self.lookup_article_matrix_[pred_input_pop], axis=2
            # )
            
           
            
            his_input_title = np.squeeze(
                self.lookup_article_matrix[his_input_title], axis=2
                )
                
            # his_input_body = np.squeeze(
            #         self.lookup_article_matrix_body[his_input_body], axis=2
            #         )
            # his_input_pop = np.squeeze(
            #         self.lookup_article_matrix[his_input_pop], axis=2
            #         )
        
       
        return (
            his_input_title,
            # his_input_body,
            # his_input_pop,
            pred_input_title,
            # pred_input_body,
            # pred_input_pop
        ), batch_y
