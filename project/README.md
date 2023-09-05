<p align="center">
    <img width="721" alt="cover-image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Tom%27s_Restaurant%2C_NYC.jpg/1280px-Tom%27s_Restaurant%2C_NYC.jpg">
</p>

## Data

There are ~$10,000$ customers in the test set. These are the customers you will need to recommend a vendor to. Each customer can order from multiple locations, identified by the variable `LOC_NUM`.

There are ~$35,000$ customers in the train set.
Some of these customers have made orders at at least one of $100$ vendors.

As said, the aim of this project is to build a recommendation engine to predict what restaurants customers are most likely to order from, given the customer location, the restaurant, and the customer order history.

Data files are available at [this link](https://mega.nz/file/ZRhgESqR#iuO6pBaZbeEttJ_BGmwbSh2XTg4tnf_zXrzSXcq5W6M) and are structured as follows:

* `test_customers.csv`- customer id’s in the test set.
* `test_locations.csv` - latitude and longitude for the different locations of each customer.
* `train_locations.csv` - customer id’s in the test set.
* `train_customers.csv` - latitude and longitude for the different locations of each customer.
* `orders.csv` - orders that the customers `train_customers.csv` from made.
* `vendors.csv` - vendors that customers can order from.
* `VariableDefinitions.txt` - Variable definitions for the datasets
* `SampleSubmission.csv` - is an example of what your submission file should look like. The order of the rows does not matter, but the names of CID X LOC_NUM X VENDOR must be correct. The column "target" is your prediction.
