__kernel void compress(__global uchar* image_in,
                       __global int* rand,
                       __global int* kmeans_centroids,
                       __global int* kmeans_px_centroids,
                       __global int* kmeans_group_size,
                       __global int* kmeans_rgb_vals,
                       int k,
                       int iter,
                       int width,
                       int height,
                       int comp)
{
    int id = get_global_id(0);

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

    int it = 0;
    while (it++ < iter)
    {
        // Iterate through each pixel in image.
        // for (int i = 0; i < width * height; i++)
        // {
        // int r_s1 = (int)(image_in[i * comp + 0]);
        // int g_s1 = (int)(image_in[i * comp + 1]);
        // int b_s1 = (int)(image_in[i * comp + 2]);
        int r_s1 = (int)(image_in[id * comp + 0]);
        int g_s1 = (int)(image_in[id * comp + 1]);
        int b_s1 = (int)(image_in[id * comp + 2]);

        float euclid = FLT_MAX;
        int group = 0;
        // Iterate through each group of k groups.
        for (int i = 0; i < k; i++)
        {
            // Find the smallest squared euclid distance to a centroid for
            // this pixel.
            int r = kmeans_centroids[i * 3 + 0] - r_s1;
            int g = kmeans_centroids[i * 3 + 1] - g_s1;
            int b = kmeans_centroids[i * 3 + 2] - b_s1;
            float e =
                sqrt(pow((float)r, 2) + pow((float)g, 2) + pow((float)b, 2));
            if (e < euclid)
            {
                euclid = e;
                group = i;
            }
        }

        // This pixel now belongs to the group with the nearest centroid.
        kmeans_px_centroids[id] = group;
        // }

        barrier(CLK_GLOBAL_MEM_FENCE);

        // Calculate the new centroid for each pixel group.
        // for (int i = 0; i < (*img)->size_pixels; i++)
        // {
        kmeans_group_size[kmeans_px_centroids[id]]++;
        kmeans_rgb_vals[kmeans_px_centroids[id] * 3 + 0] +=
            (int)(image_in[id * comp + 0]);
        kmeans_rgb_vals[kmeans_px_centroids[id] * 3 + 1] +=
            (int)(image_in[id * comp + 1]);
        kmeans_rgb_vals[kmeans_px_centroids[id] * 3 + 2] +=
            (int)(image_in[id * comp + 2]);
        // }

        barrier(CLK_GLOBAL_MEM_FENCE);

        // Average out all the pixel values.
        if (id == 0)
        {
            for (int i = 0; i < k; i++)
            {
                if (kmeans_group_size[i] == 0) continue;
                kmeans_centroids[i * 3 + 0] =
                    kmeans_rgb_vals[k * 3 + 0] / kmeans_group_size[k];
                kmeans_centroids[i * 3 + 1] =
                    kmeans_rgb_vals[k * 3 + 1] / kmeans_group_size[k];
                kmeans_centroids[i * 3 + 2] =
                    kmeans_rgb_vals[k * 3 + 2] / kmeans_group_size[k];
            }
        }
    }
}
