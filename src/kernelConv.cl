void kernel gray_conv_buff(global const uchar* image, const uint width, const uint height, global const float* mask, const uint mask_width, const uint mask_height, global uchar* out)
{
    // Mask half size
    int mask_hw = (int)floor((float)mask_width / 2);
    int mask_hh = (int)floor((float)mask_height / 2);

	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = x + y * width;

    float sum = 0.0;
    float mask_sum = 0.0;
    for (int ix = 0; ix < mask_width ; ++ix)
    {
        for (int iy = 0 ; iy < mask_height ; ++iy)
        {
            int px = x + (ix - mask_hw);
            int py = y + (iy - mask_hh);

            if (px < 0 || px >= width || py < 0 || py >= height)
            {
                continue;
            }

            // Getting mask value
            float m_value = mask[ix + iy * mask_width];
            // Getting image value
            int current_idx = px + py * width;
            sum += m_value * (float)image[current_idx];
            mask_sum += m_value;
        }
    }

	out[idx] = (uchar) floor(sum / mask_sum);
    //out[idx] = image[idx];
}

void kernel erode(global const uchar* image, 
                const uint width,
                const uint height,
                global const int* se, 
                const uint se_size,
                global uchar* out) 
{
    int hs = floor((float)se_size / 2);

    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = x + y * width;

    int current_value = (int) image[idx];

    for (int ix = 0; ix < se_size; ++ix)
    {
        for (int iy = 0 ; iy < se_size; ++iy)
        {
            int se_value = se[ix + iy * se_size];
            if (se_value == 1)
            {
                int px = x + (ix - se_size);
                int py = y + (iy - se_size);
                // Erosion
                current_value = current_value > image[px + py * width] ? image[px + py * width] : current_value;
                // Dilation
                //current_value = current_value < image[px + py * width] ? image[px + py * width] : current_value;
            }
        }
    }
    out[idx] = current_value;

    /*
    int start_x = x < hs ? x : hs;
    int start_y = y < hs ? y : hs;
    int end_x = hs + 1 > width - x ? width - x : hs + 1;
    int end_y = hs + 1 > height - y ? height - y : hs + 1;
    
    for (int ix = -start_x; ix < end_x ; ++ix)
    {
        for (int iy = -start_y ; iy < end_y ; ++iy)
        {
            int se_val = se[ix + hs + (iy + hs) * width];
        }
    }
    */
}