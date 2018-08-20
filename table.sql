create table movie (
  id int not null,
  movieid int not null default 0,
  rate int not null default 0,
  comment varchar(512) not null default '',
  primary key (id, movieid)
);