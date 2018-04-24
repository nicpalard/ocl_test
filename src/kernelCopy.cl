void kernel copy(global const uchar* image, const uint width, global uchar* out)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);
	int index = x + y * width + c;
	
	out[index] = image[index];
}