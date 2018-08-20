create table movie.movie (
  id int not null,
  movieid int not null default 0,
  rate tinyint not null default 0,
  comment varchar(640) not null default '',
  primary key (id, movieid)
)