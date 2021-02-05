#include <tensor/tensor.hpp>
#include <cmath>
#include <limits>
#include <map>

// A data structure representing a python string as storred in a numpy array (a vector of non-null terminated chars)
class pstring : public std::vector<char>
{
	public:
		pstring(const char* data, unsigned int n_chars)
		{
			this->clear();
			for (unsigned int i=0; i<n_chars; i++)
			{
				this->push_back(data[i]);
			}
		}
		friend std::ostream& operator<<(std::ostream& os, const pstring& ps)
		{
			os << "\"";
			for (unsigned int i=0; i<ps.size(); i++)
			{
				os << ps[i];
			}
			os << "\"";
			return os;
		}
};

// A template for 2D tables of data
template <typename T>
class Table : public std::vector<std::vector<T>>
{
	public:
		void print() const
		{
			//Looping on rows until no column is long enough to display anything
			bool bottom = false;
			unsigned int i = 0;
			while (!bottom)
			{
				bottom = true;
				//Looping on columns
				for (unsigned int j=0; j<this->size(); j++)
				{
					const std::vector<T>& column = this->operator[](j);
					//If the column is long enough to display the row i
					if (i < column.size())
					{
						bottom = false;
						std::cout << column[i];
					}
					std::cout << "\t";
				}
				std::cout << std::endl;
				i++;
			}
		}
};

// A function that returns a table from the data of a python numpy.ndarray of dtype=float
Table<double> as_table(double* data, unsigned int n_rows, unsigned int n_columns)
{
	//Initialize the table of data
	Table<double> table;
	table.reserve(n_columns);
	//Loop on columns
	for (unsigned int j=0; j<n_columns; j++)
	{
		//Create the column, vector to fill
		std::vector<double> column;
		column.reserve(n_rows);
		//Loop on lines
		for (unsigned int i=0; i<n_rows; i++)
		{
			column.push_back(data[i*n_columns+j]);
		}
		table.push_back(column);
	}
	return table;
}

// A function that returns two tables from the data of a python numpy.ndarray of dtype=str (codes and classes)
Table<unsigned int> as_table(char* data, unsigned int n_rows, unsigned int n_columns, unsigned int n_chars)
{
	//Initialize the table of classes, and codes associated
	Table<unsigned int> cat;
	cat.reserve(n_columns);
	//Loop on columns of the data
	for (unsigned int j=0; j<n_columns; j++)
	{
		// Create the vector of unique values encountered
		std::vector<pstring> uniques;
		uniques.reserve(n_rows);
		// Create the column vectors to fil
		std::vector<unsigned int> column;
		column.reserve(n_rows);
		// Loop on rows
		for (unsigned int i=0; i<n_rows; i++)
		{
			// get the pstring at column j and row i
			pstring string(&data[i*n_columns*n_chars+j*n_chars], n_chars);
			// Look for the index k of the pstring in the vector of uniques
			unsigned int k;
			for (k=0; k<uniques.size(); k++)
			{
				if (uniques[k] == string)
				{
					break;
				}
			}
			// if the string was never encountered, add it to 'unique'
			if (k == uniques.size())
			{
				uniques.push_back(string);
			}
			//Add the index to the column of codes
			column.push_back(k);
		}
		cat.push_back(column);
	}
	return cat;
}

//A template for comparison between a numerical values and the threshold
template <typename Tx>
bool compare(Tx x, Tx t);

template<>
bool compare<double>(double x, double t)
{
	return x <= t;
}

template <>
bool compare<unsigned int>(unsigned int x, unsigned int t)
{
	return x == t;
}

//A class reprensenting a mask for an array
class Mask
{
	public:
		Mask() {}
		Mask(unsigned int N, bool value)
		{
			n = 0;
			test = std::vector<bool>(N, value);
		}
		unsigned int size() const
		{
			return test.size();
		}
		void swap(Mask& other)
		{
			this->test.swap(other.test);
			unsigned int buffer_n = other.n;
			other.n = this->n;
			this->n = buffer_n;
		}
		bool operator[](unsigned int i) const
		{
			return test[i];
		}
		unsigned int n;
		std::vector<bool> test;
};

// Return the mask_true and mask_false for a split using a given threshold
template <typename Tx>
std::pair<Mask, Mask> split_mask(const std::vector<Tx>& x, Tx threshold, const Mask& mask, std::vector<bool>& tested)
{
	Mask mask_true(x.size(), false);
	Mask mask_false(x.size(), false);
	for (unsigned int i=0; i<mask.size(); i++)
	{
		if (!mask[i])
		{
			continue;
		}
		if (x[i] == threshold)
		{
			tested[i] = true;
		}
		if (compare<Tx>(x[i], threshold))
		{
			mask_true.test[i] = true;
			mask_true.n++;
		}
		else
		{
			mask_false.test[i] = true;
			mask_false.n++;
		}
	}
	return std::make_pair(mask_true, mask_false);
}

// Calculate the variance of the masked array
double variance(const std::vector<double>& y, const Mask& mask)
{
	unsigned int n = 0;
	double mean = 0.;
	for (unsigned int i=0; i<y.size(); i++)
	{
		//Skip masked data
		if (!mask[i])
		{
			continue;
		}
		n++;
		mean += y[i];
	}
	if (n > 0)
	{
		mean /= n;
	}
	double var = 0.;
	for (unsigned int i=0; i<y.size(); i++)
	{
		if (mask[i])
		{
			var += (y[i]-mean)*(y[i]-mean);
		}
	}
	return var;
}

// A class that represents the gain function for a given split
template <typename Ty> class Gain
{
	public:
		virtual double operator()(const std::vector<Ty>& y,
								  const Mask& mask,
								  const Mask& mask_true,
								  const Mask& mask_false) const = 0;
};

// A class that represents the gain function for a given split on categorical data
class GainCategorical : public Gain<unsigned int>
{
	public:
		virtual double measure(const std::vector<unsigned int>& y, const Mask& mask) const = 0;
		double operator()(const std::vector<unsigned int>& y,
						  const Mask& mask,
						  const Mask& mask_true,
						  const Mask& mask_false) const
		{
			double a_true = static_cast<double>(mask_true.n)/mask.n;
			double a_false = static_cast<double>(mask_false.n)/mask.n;
			return measure(y, mask) - a_true*measure(y, mask_true) - a_false*measure(y, mask_false);
		}
};

// Calculate the variance drop of a given split
class variance_drop : public Gain<double>
{
	public:
		double operator()(const std::vector<double>& y,
						  const Mask& mask,
						  const Mask& mask_true,
						  const Mask& mask_false) const
		{
			return variance(y, mask) - variance(y, mask_true) - variance(y, mask_false);
		}
};

// Calculate the frequency of occurence of each class in the masked vector, and the number of non masked entries
std::vector<double> frequencies(const std::vector<unsigned int>& y, const Mask& mask)
{
	unsigned int n = 0;
	std::vector<double> f;
	for (unsigned int i=0; i<y.size(); i++)
	{
		if (mask[i])
		{
			if (y[i]+1 > f.size())
			{
				f.resize(y[i]+1, 0.);
			}
			f[y[i]] += 1.;
			n++;
		}
	}
	for (unsigned int i=0; i<f.size(); i++)
	{
		f[i] /= n;
	}
	return f;
}

// Calculate the gini index gain of a given index
class gini_gain : public GainCategorical
{
	public:
		double measure(const std::vector<unsigned int>& y, const Mask& mask) const
		{
			std::vector<double> f = frequencies(y, mask);
			double gini = 1.;
			for (unsigned int i=0; i<f.size(); i++)
			{
				gini -= f[i]*f[i];
			}
			return gini;
		}
};

// Calculates the information (entropy) gain of a given split
class entropy_gain : public GainCategorical
{
	public:
		double measure(const std::vector<unsigned int>& y, const Mask& mask) const
		{
			std::vector<double> f = frequencies(y, mask);
			double e = 0.;
			for (unsigned int i=0; i<f.size(); i++)
			{
				if (f[i] > 0.)
				{
					e += f[i]*log2(f[i]);
				}
			}
			return e;
		}
};

// A structure representing a split of the data
struct Split
{
	Split() {}
	bool numerical_x;
	bool is_valid = false;
	unsigned int i;
	unsigned int j;
	double gain = -std::numeric_limits<double>::infinity();
	Mask mask_true;
	Mask mask_false;
};

// A function that types the split
template <typename Tx> void typing(Split& split)
{
	split.numerical_x = true;
}

// The specialization for categorical split
template<> void typing<unsigned int>(Split& split)
{
	split.numerical_x = false;
}

// Returns the best split on some data for numerical or categorical x data
template <typename Tx, typename Ty>
Split best_split_of_type(const Table<Tx>& x, const std::vector<Ty>& y, const Mask& mask, const Gain<Ty>& gain, unsigned int min_samples)
{
	Split split;
	typing<Tx>(split);
	//Looping on x columns
	for (unsigned int j=0; j<x.size(); j++)
	{
		std::vector<bool> tested(y.size(), false);
		const std::vector<Tx>& x_column = x[j];
		//Looping on unique values of the column
		for (unsigned int i=0; i<y.size(); i++)
		{
			//Skip thresholds already tested, and the masked values
			if (tested[i] || not(mask[i]))
			{
				continue;
			}
			//Split on the new threshold
			const Tx& threshold = x_column[i];
			std::pair<Mask, Mask> res = split_mask<Tx>(x_column, threshold, mask, tested);
			Mask& mask_true = res.first;
			Mask& mask_false = res.second;
			//Skip invalid splits
			if (mask_true.n < min_samples || mask_false.n < min_samples)
			{
				continue;
			}
			double g = gain(y, mask, mask_true, mask_false);
			//If the gain improved since last split
			if (g > split.gain)
			{
				split.is_valid = true;
				split.i = i;
				split.j = j;
				split.mask_true.swap(mask_true);
				split.mask_false.swap(mask_false);
				split.gain = g;
			}
		}
	}
	return split;
}

// Returns the best split on numerical and categorical x
template <typename Ty>
Split best_split(const Table<double>& numerical_x, const Table<unsigned int>& categorical_x, const std::vector<Ty>& y, const Mask& mask, const Gain<Ty>& gain, unsigned int min_samples)
{
	Split split_num_x = best_split_of_type<double>(numerical_x, y, mask, gain, min_samples);
	Split split_cat_x = best_split_of_type<unsigned int>(categorical_x, y, mask, gain, min_samples);
	if (split_num_x.gain > split_cat_x.gain)
	{
		return split_num_x;
	}
	else
	{
		return split_cat_x;
	}
}

// A struct representing a branch
struct Branch
{
	Branch() {}
	Branch(unsigned int N)
	{
		mask = Mask(N, true);
	}
	template <typename Ty> Ty predicted();
	Split split; //Split performed on the data of this branch
	Mask mask; //Mask of the data that reaches this branch
	unsigned int child_true = 0; //index in the tree of branche for the childs that verifies the test
	unsigned int child_false = 0; //... for the childs that don't verify the test
	unsigned int depth = 0; //depth of the branch in the tree
	double y_num; //Numerical prediction of the leaf
	unsigned int iy_cat; //Index in y of the categorical prediction of the leaf
};

template<>
unsigned int Branch::predicted<unsigned int>()
{
	return iy_cat;
}

template<>
double Branch::predicted<double>()
{
	return y_num;
}

// Split a branch
template <typename Ty>
void split_branch(std::vector<Branch>& branches, std::list<unsigned int>& leaf_indexes, unsigned int index,
				  const Table<double>& numerical_x, const Table<unsigned int>& categorical_x, const std::vector<Ty>& y, const Gain<Ty>& gain,
				  unsigned int min_samples, unsigned int max_depth)
{
	leaf_indexes.remove(index);
	Branch& branch = branches[index];
	Split& split = branch.split;
	unsigned int depth = branch.depth+1;
	// Adding the leaf_true
	Branch child_true;
	child_true.mask = split.mask_true;
	child_true.depth = depth;
	branch.child_true = branches.size();
	leaf_indexes.push_back(branch.child_true);
	// Adding the leaf_false
	Branch child_false;
	child_false.mask = split.mask_false;
	child_false.depth = depth;
	branch.child_false = branches.size()+1;
	leaf_indexes.push_back(branch.child_false);
	// Calculating the gain of the new leafs
	if (depth < max_depth)
	{
		child_true.split = best_split<Ty>(numerical_x, categorical_x, y, child_true.mask, gain, min_samples);
		child_false.split = best_split<Ty>(numerical_x, categorical_x, y, child_false.mask, gain, min_samples);
	}
	// Adding the childs in the tree
	branches.push_back(child_true);
	branches.push_back(child_false);
}

// A struct representing a branche in c
struct cBranch
{
	bool is_leaf = false; //True if the branch is a leaf of the tree
	unsigned int child_true; //index in the tree of branche for the childs that verifies the test
	unsigned int child_false; //... for the childs that don't verify the test
	double gain; //Gain of the split performed on this branch
	bool split_numerical_x; //If true the split must be performed on numerical x (otherwise categorical x)
	unsigned int i; //Line of the threshold used for the split
	unsigned int j; //Column of the threshold used for the split
	double y_num; //Numerical prediction of the leaf
	unsigned int iy_cat; //Index in y of the categorical prediction of the leaf
};

// A struct representing a tree
struct cTree
{
	unsigned int n_branches;
	cBranch* branches = nullptr;
};


// A function to obtain the leaf prediction from the data that fell in the leaf
template <typename Ty>
void leaf_prediction(const std::vector<Ty>& y, Branch& branch);

template<>
void leaf_prediction<double>(const std::vector<double>& y, Branch& branch)
{
	std::vector<unsigned int> classes;

	unsigned int n = 0;
	branch.y_num = 0.;
	for (unsigned int i=0; i<y.size(); i++)
	{
		// skip masked data
		if (!branch.mask[i])
		{
			continue;
		}
		// calculate the mean
		branch.y_num += y[i];
		n ++;
	}
	branch.y_num /= n;
}

template<>
void leaf_prediction<unsigned int>(const std::vector<unsigned int>& y, Branch& branch)
{
	std::map<unsigned int, unsigned int> occurences;
	std::map<unsigned int, unsigned int> index;
	// Count occurence of each category for y that falls in the leaf
	for (unsigned int i=0; i<y.size(); i++)
	{
		// skip masked data
		if (!branch.mask[i])
		{
			continue;
		}
		unsigned int cat = y[i];
		// Increase the occurence counter of the given category
		if (occurences.count(cat) > 0)
		{
			occurences[cat] ++;
		}
		else
		{
			occurences[cat] = 1;
			//index of first occurence of the category
			index[cat] = i;
		}
	}
	// Find most frequent occurence of a category
	unsigned int n_max = 0;
	unsigned int cat_max = 0;
	for (std::pair<unsigned int, unsigned int> occ : occurences)
	{
		if (occ.second > n_max)
		{
			n_max = occ.second;
			cat_max = occ.first;
		}
	}
	branch.iy_cat = cat_max;
}

// Grow a decision tree
template <typename Ty>
cTree grow_tree(const Table<double>& num_x, const Table<unsigned int>& cat_x, const Table<Ty> Y, const Gain<Ty>& gain, unsigned int min_samples, unsigned int max_depth, unsigned int max_leafs, Ty* y_predicted)
{
	const std::vector<Ty>& y = Y[0];
	Branch root(y.size());
	root.split = best_split<Ty>(num_x, cat_x, y, root.mask, gain, min_samples);
	std::vector<Branch> branches = {root};
	std::list<unsigned int> leaf_indexes = {0};
	bool can_split = true;
	while(can_split)
	{
		can_split = false;
		// If there is too much leafs, breaks
		if (leaf_indexes.size() >= max_leafs)
		{
			break;
		}
		// Look for the the best leaf to split
		double best_gain = -std::numeric_limits<double>::infinity();
		unsigned int l_best = 0;
		for (unsigned int l : leaf_indexes)
		{
			//Skip invalid splits
			if (!branches[l].split.is_valid)
			{
				continue;
			}
			can_split = true;
			// Test if this leaf is the new best leaf to split
			if (branches[l].split.gain > best_gain)
			{
				best_gain = branches[l].split.gain;
				l_best = l;
			}
		}
		// Split the leaf if possible
		if (can_split)
		{
			split_branch(branches, leaf_indexes, l_best, num_x, cat_x, y, gain, min_samples, max_depth);
		}
	}
	// Setting predicted value for all leafs
	for (unsigned int i_leaf : leaf_indexes)
	{
		leaf_prediction<Ty>(y, branches[i_leaf]);
	}
	// Saving the predicted values
	if (y_predicted != nullptr)
	{
		for (unsigned int i=0; i<y.size(); i++)
		{
			for (unsigned int l : leaf_indexes)
			{
				if (branches[l].mask[i])
				{
					y_predicted[i] = branches[l].predicted<Ty>();
					break;
				}
			}
		}
	}
	// Filling the cTree
	cTree tree;
	unsigned int n = branches.size();
	tree.n_branches = n;
	tree.branches = new cBranch[n];
	for (unsigned int i=0; i<n; i++)
	{
		tree.branches[i].is_leaf = branches[i].child_true == 0;
		tree.branches[i].child_true = branches[i].child_true;
		tree.branches[i].child_false = branches[i].child_false;
		tree.branches[i].gain = branches[i].split.gain;
		tree.branches[i].split_numerical_x = branches[i].split.numerical_x;
		tree.branches[i].i = branches[i].split.i;
		tree.branches[i].j = branches[i].split.j;
		tree.branches[i].y_num = branches[i].y_num;
		tree.branches[i].iy_cat = branches[i].iy_cat;
	}
	return tree;
}

//////////////////////////////////////////////////////////////////////////////////////////

// Grow a classifier tree
extern "C" cTree grow_classifier(double* numerical_x, char* categorical_x, char* y, unsigned int n_rows, unsigned int n_numerical_columns, unsigned int n_categorical_columns, unsigned int n_chars_x, unsigned int n_chars_y, unsigned int min_samples, unsigned int max_depth, unsigned int max_leafs, bool gini, unsigned int* y_predicted)
{
	Table<double> num_x = as_table(numerical_x, n_rows, n_numerical_columns);
	Table<unsigned int> cat_x = as_table(categorical_x, n_rows, n_categorical_columns, n_chars_x);
	Table<unsigned int> Y = as_table(y, n_rows, 1, n_chars_y);
	if (gini)
	{
		return grow_tree<unsigned int>(num_x, cat_x, Y, gini_gain(), min_samples, max_depth, max_leafs, y_predicted);
	}
	else
	{
		return grow_tree<unsigned int>(num_x, cat_x, Y, entropy_gain(), min_samples, max_depth, max_leafs, y_predicted);
	}
}

// Grow a regressor tree
extern "C" cTree grow_regressor(double* numerical_x, char* categorical_x, double* y, unsigned int n_rows, unsigned int n_numerical_columns, unsigned int n_categorical_columns, unsigned int n_chars_x, unsigned int min_samples, unsigned int max_depth, unsigned int max_leafs, double* y_predicted)
{
	Table<double> num_x = as_table(numerical_x, n_rows, n_numerical_columns);
	Table<unsigned int> cat_x = as_table(categorical_x, n_rows, n_categorical_columns, n_chars_x);
	Table<double> Y = as_table(y, n_rows, 1);
	return grow_tree<double>(num_x, cat_x, Y, variance_drop(), min_samples, max_depth, max_leafs, y_predicted);
}

// Delete a tree
extern "C" void delete_tree(cTree tree)
{
	delete tree.branches;
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
//Entry point for debug
int main(int argc, char* argv[])
{
	double num_x[] = {1., 2., 3., 4.,
					  5., 6., 7., 8.,
					  9., 10., 11., 12.};
	char cat_x[] = {};
	double y[] = {10., 10., 100.};
	unsigned int n_chars_x = 0;
	unsigned int n_rows = 3;
	unsigned int n_numerical_columns = 4;
	unsigned int n_categorical_columns = 0;
	cTree tree = grow_regressor(num_x, cat_x, y, n_rows, n_numerical_columns, n_categorical_columns, n_chars_x, 1, 1, 2, nullptr);
	delete_tree(tree);
}
*/
