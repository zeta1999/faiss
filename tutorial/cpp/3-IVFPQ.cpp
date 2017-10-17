/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>


int main() {
    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 10000;                        // nb of queries

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }


    // I believe this is the number of inverted lists to create? That is, how
    // many clusters to divide the dataset into?
    int nlist = 100;

    // It looks like 'k' is just a search-time paramter--dictating how many
    // nearest neighbors to return.
    int k = 4;

    // 'm' is the number of subsections to divide the vectors into.
    int m = 8;
    faiss::IndexFlatL2 quantizer(d);       // the other index

    // The final value '8' is the number of bits to use for the codes, which
    // dictates the number of clusters to learn for each subvector.
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);

    // here we specify METRIC_L2, by default it performs inner-product search

    // 'train' is a method of the ProductQuantizer object.
    index.train(nb, xb);
    index.add(nb, xb);

    {       // sanity check
        // 'I' holds the indeces for the five queries and 'D' holds the
        // distances.
        long *I = new long[k * 5];
        float *D = new float[k * 5];

        index.search(5, xb, k, D, I);

        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }



    delete [] xb;
    delete [] xq;

    return 0;
}
