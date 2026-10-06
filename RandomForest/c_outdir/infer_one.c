#include "infer_one.h"

// Returns prediction for one datapoint
int infer_one(const int *x)
{
    int votes[TREES];
    for (int i = 0; i < TREES; i++)
    	votes[i] = 0;
    int vote_count = 0;

    int row = 0;

    while (trees[row][0] != 0 || trees[row][1] != 0 ||
           trees[row][2] != 0 || trees[row][3] != 0)
    {
        int feature = trees[row][0];
        int value   = trees[row][1];

        // leaf?
        if (value == -1)
        {
            int cls = (int)feature;
            votes[vote_count++] = cls;

            // skip to next tree
            while (!(trees[row][0] == 0 && trees[row][1] == 0 &&
                     trees[row][2] == 0 && trees[row][3] == 0))
                row++;

            // skip the {0,0,0,0}
            row++;
            continue;
        }

        int fidx = (int)feature;
        int split = value;
        int left  = (int)trees[row][2];
        int right = (int)trees[row][3];

        if (x[fidx] < split)
            row = left + row;
        else
            row = right + row;
    }

    // Majority vote
    int best_class = -1, best_count = 0;
    for (int i = 0; i < CLASSES; i++)
    {
        int c = 0;
        for (int j = 0; j < vote_count; j++)
            if (votes[j] == i) c++;

        if (c > best_count)
        {
            best_class = i;
            best_count = c;
        }
    }

    return best_class;
}
