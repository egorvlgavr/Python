# Analyzing of click stream and training set data
## Description
This program analyse typical e-commerce click stream data and training set in RankLib format.
Data is read line by line so huge files so it can be used for analysis of huge files.
There are several available modes:
 * Statistic
 * Queries
 * Histogram

Program settings are placed in **config.json**.
#### Statistic
Apply only for click stream.
Print: number of query - product pairs, number of unique queries, max/min/avg products per query, products with zero clicks
#### Queries
Apply only for click stream.
Write in file all unique queries that was filtered by several filters. Can add to queries number of products.
#### Histogram
Apply for click stream and training set
Show histogram of click stream/training set column data. 
## Information
This program is written on *Python 3.5*. It use **matplotlib** for visualisation of data. 
