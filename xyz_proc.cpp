#include <opencv2/highgui/highgui.hpp>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/recognition/linemod/line_rgbd.h>

using namespace std;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointXY Point2D;
typedef pcl::PointCloud<Point2D> PointCloud2D;

/*
void ViewSignleCloud(PointCloud::Ptr cloud)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	//viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud);
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void ViewDualCloud(PointCloud::Ptr cloud1, PointCloud::Ptr cloud2)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->addPointCloud<Point>(cloud1, "cloud 1", v1);
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->addPointCloud<Point>(cloud2, "cloud 2", v2);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
*/

//Read xyz file
bool ReadXYZ(const string &scan_name, PointCloud::Ptr cloud)
{
	ifstream fs;
	fs.open(scan_name.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail())
	{
		PCL_ERROR("Could not open file '%s'! Error : %s\n", scan_name.c_str(), strerror(errno));
		fs.close();
		return false;
	}

	std::string line;
	std::vector<std::string> st;
	while (!fs.eof())
	{
		getline(fs, line);
		// Ignore empty lines
		if (line == "")
			continue;
		// Tokenize the line
		boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r "), boost::token_compress_on);
		if (st.size() != 3)
			continue;
		cloud->push_back(Point(float(atof(st[0].c_str())), float(atof(st[1].c_str())), float(atof(st[2].c_str()))));
	}
	fs.close();

	cloud->width = uint32_t(cloud->size());
	cloud->height = 1;
	cloud->is_dense = true;
	return true;
}

//Write xyz file
void WriteXYZ(string scan_name, PointCloud::Ptr cloud)
{
  ofstream writeFile;
  writeFile.open(scan_name.c_str());
  for (size_t p = 0; p < cloud->points.size(); p++)
  {
    writeFile << cloud->points[p].x << " ";
    writeFile << cloud->points[p].y << " ";
    writeFile << cloud->points[p].z << "\n";
  }
  writeFile.close();
}

//Split the cloud if it contains multiple sherds (the cloud is supposed to be scanned horizontally)
//@in_file: the input xyz file
//@out_dir: the directory of output split xyz files
//@min_height_percent: erase points under a certain height
//@min_size_percent: ignore point clusters smaller than a certain size
extern "C"
int SplitCloud(char* in_dir, char* scan_name, char* out_dir, double min_height_percent, double min_size_percent)
{
	PointCloud::Ptr src_cloud(new PointCloud);
	string in_dir_str(in_dir);
	string scan_name_str(scan_name);
	bool if_read = ReadXYZ(in_dir_str + '/' + scan_name_str, src_cloud);
	if (!if_read)
		return -1;
	else
	{
		PointCloud::Ptr all_sherd_cloud(new PointCloud);
		Point min_bound, max_bound;
		pcl::getMinMax3D(*src_cloud, min_bound, max_bound);
		double height_thre = (max_bound.z - min_bound.z) * min_height_percent;
		//double height_thre = 5;
		for (size_t i = 0; i < src_cloud->points.size(); i++)
		{
			if (src_cloud->points[i].z - min_bound.z > height_thre)
				all_sherd_cloud->points.push_back(src_cloud->points[i]);
		}

		//split the cloud by KD-Tree clustering
		pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
		tree->setInputCloud(all_sherd_cloud);
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance(1);//If the gap is greater than 1mm, split the cloud
		int size_thre = src_cloud->points.size() * min_size_percent;
		ec.setMinClusterSize(size_thre);
		ec.setMaxClusterSize(all_sherd_cloud->points.size());
		ec.setSearchMethod(tree);
		ec.setInputCloud(all_sherd_cloud);
		ec.extract(cluster_indices);

		//create new point clouds
		vector<PointCloud::Ptr> split_cloud_vec;
		for (size_t c = 0; c < cluster_indices.size(); c++)
		{
			PointCloud::Ptr temp_sherd_cloud(new PointCloud);
			for (size_t p = 0; p < cluster_indices[c].indices.size(); p++)
			{
				int temp_index = cluster_indices[c].indices[p];
				temp_sherd_cloud->points.push_back(all_sherd_cloud->points[temp_index]);
			}
			split_cloud_vec.push_back(temp_sherd_cloud);
		}

		//write split clouds to xyz files
		string out_dir_str(out_dir);
		scan_name_str = scan_name_str.substr(0, scan_name_str.size()-4);
		for (size_t c = 0; c < split_cloud_vec.size(); c++)
		{
			char idx = '1' + c;
			string sub_out_dir = out_dir_str + '/' + scan_name_str + '-' + idx;
			int status = mkdir(sub_out_dir.c_str(), 0777);
			string out_path = sub_out_dir + "/sherd.xyz";
			WriteXYZ(out_path, split_cloud_vec[c]);
		}
		return split_cloud_vec.size();
	}
}

// //Mapping 3D cloud to 2D Matrix
// void CloudSampling(PointCloud::Ptr cloud, cv::Mat &depth_mat, double resolution)
// {
// 	Point min_bound, max_bound;
// 	pcl::getMinMax3D(*cloud, min_bound, max_bound);
// 	double cloud_width = max_bound.x - min_bound.x;
// 	double cloud_height = max_bound.y - min_bound.y;
// 	int image_width = (int)(cloud_width / resolution);
// 	int image_height = (int)(cloud_height / resolution);
// 	depth_mat = cv::Mat(image_height, image_width, CV_32F, cv::Scalar(-1)); //-1 means background
//
// 	PointCloud2D::Ptr cloud2D(new PointCloud2D);
// 	for (size_t i = 0; i < cloud->points.size(); i++)
// 	{
// 		Point2D tempPoint{ cloud->points[i].x - min_bound.x, cloud->points[i].y - min_bound.y };
// 		cloud2D->points.push_back(tempPoint);
// 	}
//
// 	pcl::KdTreeFLANN<pcl::PointXY> kdtree;
// 	kdtree.setInputCloud(cloud2D);
// 	int K = 1;
// 	for (size_t i = 0; i < image_height; i++)
// 	{
// 		for (size_t j = 0; j < image_width; j++)
// 		{
// 			float x = j * resolution;
// 			float y = i * resolution;
// 			std::vector<int> pointIdxNKNSearch(K);
// 			std::vector<float> pointNKNSquaredDistance(K);
// 			kdtree.nearestKSearch(Point2D{ x, y }, K, pointIdxNKNSearch, pointNKNSquaredDistance);
// 			if (pointNKNSquaredDistance[0] < resolution)
// 			{
// 				float temp_depth = cloud->points[pointIdxNKNSearch[0]].z;
// 				depth_mat.at<float>(i, j) = temp_depth - min_bound.z;
// 			}
// 		}
// 	}
// }
//
// //Normalize float matrix to image (0-255), and create a mask image (background)
// void NormDepthImg(cv::Mat &src_img, cv::Mat &mask_img, cv::Mat &dst_img)
// {
// 	float min = FLT_MAX, max = FLT_MIN;
// 	mask_img = cv::Mat(src_img.rows, src_img.cols, CV_8U, cv::Scalar(255));
// 	for (size_t i = 0; i < src_img.rows; i++)
// 	{
// 		for (size_t j = 0; j < src_img.cols; j++)
// 		{
// 			float temp = src_img.at<float>(i, j);
// 			if (temp == -1)
// 				mask_img.at<uchar>(i, j) = 0;
// 			else if (temp > max)
// 				max = temp;
// 			else if (temp < min)
// 				min = temp;
// 		}
// 	}
// 	dst_img = cv::Mat(src_img.rows, src_img.cols, CV_8U, cv::Scalar(0));
// 	for (size_t i = 0; i < src_img.rows; i++)
// 	{
// 		for (size_t j = 0; j < src_img.cols; j++)
// 		{
// 			if (mask_img.at<uchar>(i, j) == 255)
// 			{
// 				int temp = (int)((src_img.at<float>(i, j) - min) * 255 / (max - min));
// 				dst_img.at<uchar>(i, j) = temp; //high gray value means high elevation
// 			}
// 		}
// 	}
// }
//
// //Convert xyz file to depth image
// extern "C"
// int xyz2depth(char* xyz_name, char* depth_name, char* mask_name, double resolution)
// {
// 	PointCloud::Ptr src_cloud(new PointCloud);
// 	bool if_read = ReadXYZ(xyz_name, src_cloud);
//   if (!if_read)
//     return -1;
//   else
//   {
//     cv::Mat depth_mat;
//     CloudSampling(src_cloud, depth_mat, resolution);
//   	cv::flip(depth_mat, depth_mat, 0);
// 		cv::Mat depth_img, mask_img;
// 		NormDepthImg(depth_mat, mask_img, depth_img);
//     //cv::imshow("1", depth_img);
//     //cv::waitKey();
//     cv::imwrite(depth_name, depth_img);
//     cv::imwrite(mask_name, mask_img);
//   }
//   return 1;
// }


/*
int main(int argc, char *argv[])
{
	PointCloud::Ptr src_cloud(new PointCloud);
	bool if_read = ReadXYZ(argv[1], src_cloud);
  if (!if_read)
    return -1;
  else
  {
    double sample_resolution = 0.1;
    cv::Mat depth_mat;
    CloudSampling(src_cloud, depth_mat, sample_resolution);
  	cv::flip(depth_mat, depth_mat, 0);
    cout << depth_mat.cols << endl;
		cv::Mat depth_img, mask_img;
		NormDepthImg(depth_mat, mask_img, depth_img);
    //cv::imshow("1", depth_img);
    //cv::waitKey();
    cv::imwrite(argv[2], depth_img);
    cout << "haha" << endl;
  }
  return 0;
}
*/
