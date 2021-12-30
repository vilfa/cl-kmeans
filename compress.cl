__kernel void kmeans(__global unsigned char* image_in,
                     __global int* kmeans_centroids,
                     __global unsigned int* kmeans_px_centroids,
                     __global int* rand,
                     int k,
                     int iter,
                     int width,
                     int height,
                     int comp)
{
    int id = get_global_id(0);

    int x = id / width;
    int y = id % width;

    if (id > width * height) return;

    if (id == 0)
    {
        for (int i = 0; i < k; i++)
        {
            kmeans_centroids[i * 3 + 0] = image_in[rand[i] * comp + 0];
            kmeans_centroids[i * 3 + 1] = image_in[rand[i] * comp + 1];
            kmeans_centroids[i * 3 + 2] = image_in[rand[i] * comp + 2];
        }
    }

    int i = 0;
    while (i++ < iter)
    {
        // printf("processing iteration %d/%d...\n", iter, (*kmn)->iter);

        kmean_sample_t sample;

        // Iterate through each pixel in image.
        double euclid;
        uint32_t group;
        for (int i = 0; i < (*img)->size_pixels; i++)
        {
            sample.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
            sample.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
            sample.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

            euclid = DBL_MAX;
            group = 0;
            // Iterate through each group of k groups.
            for (uint32_t k = 0; k < (*kmn)->k; k++)
            {
                // Find the smallest squared euclid distance to a centroid for
                // this pixel.
                double e =
                    kmeans_sample_euclid2(&((*kmn)->centroids[k]), &sample);
                if (e < euclid)
                {
                    euclid = e;
                    group = k;
                }
            }

            // This pixel now belongs to the group with the nearest centroid.
            (*kmn)->px_centroid[i] = group;
        }

        // Calculate the new centroid for each pixel group.
        uint32_t* group_size = (uint32_t*)calloc((*kmn)->k, sizeof(uint32_t));
        int* rgb_vals = (int*)calloc(3 * (*kmn)->k, sizeof(int));

        for (int i = 0; i < (*img)->size_pixels; i++)
        {
            group_size[(*kmn)->px_centroid[i]]++;
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 0] +=
                (int)((*img)->DATA[i * (*img)->comp + 0]);
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 1] +=
                (int)((*img)->DATA[i * (*img)->comp + 1]);
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 2] +=
                (int)((*img)->DATA[i * (*img)->comp + 2]);
        }

        // Average out all the pixel values.
        for (uint32_t k = 0; k < (*kmn)->k; k++)
        {
            if (group_size[k] == 0) continue;
            (*kmn)->centroids[k].r = rgb_vals[k * 3 + 0] / group_size[k];
            (*kmn)->centroids[k].g = rgb_vals[k * 3 + 1] / group_size[k];
            (*kmn)->centroids[k].b = rgb_vals[k * 3 + 2] / group_size[k];
            (*kmn)->centroids[k].norm =
                kmeans_sample_norm((&(*kmn)->centroids[k]));
        }

        free(group_size);
        free(rgb_vals);
    }
}
