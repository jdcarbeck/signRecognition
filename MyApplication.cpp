#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <cmath>
#include <experimental/filesystem> // C++-standard header file name
// #include <filesystem> // Microsoft-specific implementation header file name

using namespace std::experimental::filesystem::v1;
using namespace std;
using namespace cv::ml;

// Sign must be at least 100x100
#define MINIMUM_SIGN_SIDE 100
#define MINIMUM_SIGN_AREA 10000
#define MINIMUM_SIGN_BOUNDARY_LENGTH 400
#define STANDARD_SIGN_WIDTH_AND_HEIGHT 200
// Best match must be 10% better than second best match
#define REQUIRED_RATIO_OF_BEST_TO_SECOND_BEST 1.1
// Located shape must overlap the ground truth by 80% to be considered a match
#define REQUIRED_OVERLAP 0.8

class ObjectAndLocation
{
public:
	ObjectAndLocation(string object_name, Point top_left, Point top_right, Point bottom_right, Point bottom_left, Mat object_image);
	ObjectAndLocation(FileNode &node);
	void write(FileStorage &fs);
	void read(FileNode &node);
	Mat &getImage();
	string getName();
	void setName(string new_name);
	string getVerticesString();
	void DrawObject(Mat *display_image, Scalar &colour);
	double getMinimumSideLength();
	double getArea();
	void getVertice(int index, int &x, int &y);
	void setImage(Mat image);							   // *** Student should add any initialisation (of their images or features; see private data below) they wish into this method.
	double compareObjects(ObjectAndLocation *otherObject); // *** Student should write code to compare objects using chosen method.
	bool OverlapsWith(ObjectAndLocation *other_object);
	void FeatureVector(vector<Point2f> &features);
	void setFeatures(vector<Point2f> features);
	vector<Point2f> getFeatures();

private:
	string object_name;
	Mat image;
	vector<Point2i> vertices;
	vector<Point2f> featureVector;
	// *** Student can add whatever images or features they need to describe the object.
};

class AnnotatedImages;

class ImageWithObjects
{
	friend class AnnotatedImages;

public:
	ImageWithObjects(string passed_filename);
	ImageWithObjects(FileNode &node);
	virtual void LocateAndAddAllObjects(AnnotatedImages &training_images) = 0;
	ObjectAndLocation *addObject(string object_name, int top_left_column, int top_left_row, int top_right_column, int top_right_row,
								 int bottom_right_column, int bottom_right_row, int bottom_left_column, int bottom_left_row, Mat &image);
	void write(FileStorage &fs);
	void read(FileNode &node);
	int objectCount();
	ObjectAndLocation *getObject(int index);
	void extractAndSetObjectImage(ObjectAndLocation *new_object);
	string ExtractObjectName(string filenamestr);
	void FindBestMatch(ObjectAndLocation *new_object, string &object_name, double &match_value);

protected:
	string filename;
	Mat image;
	vector<ObjectAndLocation> objects;
};

class ImageWithBlueSignObjects : public ImageWithObjects
{
public:
	ImageWithBlueSignObjects(string passed_filename);
	ImageWithBlueSignObjects(FileNode &node);
	void LocateAndAddAllObjects(AnnotatedImages &training_images); // *** Student needs to develop this routine and add in objects using the addObject method
	void AddSignObject(Mat img, Rect crop);
};

class ConfusionMatrix;

class AnnotatedImages
{
public:
	AnnotatedImages(string directory_name);
	AnnotatedImages();
	void addAnnotatedImage(ImageWithObjects &annotated_image);
	void write(FileStorage &fs);
	void read(FileStorage &fs);
	void read(FileNode &node);
	void read(string filename);
	void LocateAndAddAllObjects(AnnotatedImages &training_images);
	void FindBestMatch(ObjectAndLocation *new_object);
	Mat getImageOfAllObjects(int break_after = 7);
	void CompareObjectsWithGroundTruth(AnnotatedImages &training_images, AnnotatedImages &ground_truth, ConfusionMatrix &results);
	ImageWithObjects *getAnnotatedImage(int index);
	ImageWithObjects *FindAnnotatedImage(string filename_to_find);

public:
	string name;
	vector<ImageWithObjects *> annotated_images;
};

class ConfusionMatrix
{
public:
	ConfusionMatrix(AnnotatedImages training_images);
	void AddMatch(string ground_truth, string recognised_as, bool duplicate = false);
	void AddFalseNegative(string ground_truth);
	void AddFalsePositive(string recognised_as);
	void Print();

private:
	void AddObjectClass(string object_class_name);
	int getObjectClassIndex(string object_class_name);
	vector<string> class_names;
	int confusion_size;
	int **confusion_matrix;
	int false_index;
	int tp, fp, fn;
};

ObjectAndLocation::ObjectAndLocation(string passed_object_name, Point top_left, Point top_right, Point bottom_right, Point bottom_left, Mat object_image)
{
	object_name = passed_object_name;
	vertices.push_back(top_left);
	vertices.push_back(top_right);
	vertices.push_back(bottom_right);
	vertices.push_back(bottom_left);
	setImage(object_image);
}
ObjectAndLocation::ObjectAndLocation(FileNode &node)
{
	read(node);
}
void ObjectAndLocation::write(FileStorage &fs)
{
	fs << "{"
	   << "nameStr" << object_name;
	fs << "coordinates"
	   << "[";
	for (int i = 0; i < vertices.size(); ++i)
	{
		fs << "[:" << vertices[i].x << vertices[i].y << "]";
	}
	fs << "]";
	fs << "}";
}
void ObjectAndLocation::read(FileNode &node)
{
	node["nameStr"] >> object_name;
	FileNode data = node["coordinates"];
	for (FileNodeIterator itData = data.begin(); itData != data.end(); ++itData)
	{
		// Read each point
		FileNode pt = *itData;

		Point2i point;
		FileNodeIterator itPt = pt.begin();
		point.x = *itPt;
		++itPt;
		point.y = *itPt;
		vertices.push_back(point);
	}
}
Mat &ObjectAndLocation::getImage()
{
	return image;
}
string ObjectAndLocation::getName()
{
	return object_name;
}
void ObjectAndLocation::setName(string new_name)
{
	object_name.assign(new_name);
}
string ObjectAndLocation::getVerticesString()
{
	string result;
	for (int index = 0; (index < vertices.size()); index++)
		result.append("(" + to_string(vertices[index].x) + " " + to_string(vertices[index].y) + ") ");
	return result;
}
void ObjectAndLocation::DrawObject(Mat *display_image, Scalar &colour)
{
	writeText(*display_image, object_name, vertices[0].y - 8, vertices[0].x + 8, colour, 2.0, 4);
	polylines(*display_image, vertices, true, colour, 8);
}
double ObjectAndLocation::getMinimumSideLength()
{
	double min_distance = DistanceBetweenPoints(vertices[0], vertices[vertices.size() - 1]);
	for (int index = 0; (index < vertices.size() - 1); index++)
	{
		double distance = DistanceBetweenPoints(vertices[index], vertices[index + 1]);
		if (distance < min_distance)
			min_distance = distance;
	}
	return min_distance;
}
double ObjectAndLocation::getArea()
{
	return contourArea(vertices);
}
void ObjectAndLocation::getVertice(int index, int &x, int &y)
{
	if ((vertices.size() < index) || (index < 0))
		x = y = -1;
	else
	{
		x = vertices[index].x;
		y = vertices[index].y;
	}
}

ImageWithObjects::ImageWithObjects(string passed_filename)
{
	filename = strdup(passed_filename.c_str());
	cout << "Opening " << filename << endl;
	image = imread(filename, -1);
}
ImageWithObjects::ImageWithObjects(FileNode &node)
{
	read(node);
}
ObjectAndLocation *ImageWithObjects::addObject(string object_name, int top_left_column, int top_left_row, int top_right_column, int top_right_row,
											   int bottom_right_column, int bottom_right_row, int bottom_left_column, int bottom_left_row, Mat &image)
{
	ObjectAndLocation new_object(object_name, Point(top_left_column, top_left_row), Point(top_right_column, top_right_row), Point(bottom_right_column, bottom_right_row), Point(bottom_left_column, bottom_left_row), image);
	objects.push_back(new_object);
	return &(objects[objects.size() - 1]);
}
void ImageWithObjects::write(FileStorage &fs)
{
	fs << "{"
	   << "Filename" << filename << "Objects"
	   << "[";
	for (int index = 0; index < objects.size(); index++)
		objects[index].write(fs);
	fs << "]"
	   << "}";
}
void ImageWithObjects::extractAndSetObjectImage(ObjectAndLocation *new_object)
{
	Mat perspective_warped_image = Mat::zeros(STANDARD_SIGN_WIDTH_AND_HEIGHT, STANDARD_SIGN_WIDTH_AND_HEIGHT, image.type());
	Mat perspective_matrix(3, 3, CV_32FC1);
	int x[4], y[4];
	new_object->getVertice(0, x[0], y[0]);
	new_object->getVertice(1, x[1], y[1]);
	new_object->getVertice(2, x[2], y[2]);
	new_object->getVertice(3, x[3], y[3]);
	Point2f source_points[4] = {{((float)x[0]), ((float)y[0])}, {((float)x[1]), ((float)y[1])}, {((float)x[2]), ((float)y[2])}, {((float)x[3]), ((float)y[3])}};
	Point2f destination_points[4] = {{0.0, 0.0}, {STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, 0.0}, {STANDARD_SIGN_WIDTH_AND_HEIGHT - 1, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1}, {0.0, STANDARD_SIGN_WIDTH_AND_HEIGHT - 1}};
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);
	warpPerspective(image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());
	new_object->setImage(perspective_warped_image);
}
void ImageWithObjects::read(FileNode &node)
{
	filename = (string)node["Filename"];
	image = imread(filename, -1);
	FileNode images_node = node["Objects"];
	if (images_node.type() == FileNode::SEQ)
	{
		for (FileNodeIterator it = images_node.begin(); it != images_node.end(); ++it)
		{
			FileNode current_node = *it;
			ObjectAndLocation *new_object = new ObjectAndLocation(current_node);
			extractAndSetObjectImage(new_object);
			objects.push_back(*new_object);
		}
	}
}
ObjectAndLocation *ImageWithObjects::getObject(int index)
{
	if ((index < 0) || (index >= objects.size()))
		return NULL;
	else
		return &(objects[index]);
}
int ImageWithObjects::objectCount()
{
	return objects.size();
}
void ImageWithObjects::FindBestMatch(ObjectAndLocation *new_object, string &object_name, double &match_value)
{
	for (int index = 0; (index < objects.size()); index++)
	{

		vector<Point2f> features;
		objects[index].FeatureVector(features);
		objects[index].setFeatures(features);
		double temp_match_score = objects[index].compareObjects(new_object);
		if ((temp_match_score > 0.0) && ((match_value < 0.0) || (temp_match_score < match_value)))
		{
			object_name = objects[index].getName();
			match_value = temp_match_score;
		}
	}
}

string ImageWithObjects::ExtractObjectName(string filenamestr)
{
	int last_slash = filenamestr.rfind("/");
	int start_of_object_name = (last_slash == std::string::npos) ? 0 : last_slash + 1;
	int extension = filenamestr.find(".", start_of_object_name);
	int end_of_filename = (extension == std::string::npos) ? filenamestr.length() - 1 : extension - 1;
	int end_of_object_name = filenamestr.find_last_not_of("1234567890", end_of_filename);
	end_of_object_name = (end_of_object_name == std::string::npos) ? end_of_filename : end_of_object_name;
	string object_name = filenamestr.substr(start_of_object_name, end_of_object_name - start_of_object_name + 1);
	return object_name;
}

ImageWithBlueSignObjects::ImageWithBlueSignObjects(string passed_filename) : ImageWithObjects(passed_filename)
{
}
ImageWithBlueSignObjects::ImageWithBlueSignObjects(FileNode &node) : ImageWithObjects(node)
{
}

AnnotatedImages::AnnotatedImages(string directory_name)
{
	name = directory_name;
	for (std::experimental::filesystem::directory_iterator next(std::experimental::filesystem::path(directory_name.c_str())), end; next != end; ++next)
	{
		read(next->path().generic_string());
	}
}
AnnotatedImages::AnnotatedImages()
{
	name = "";
}
void AnnotatedImages::addAnnotatedImage(ImageWithObjects &annotated_image)
{
	annotated_images.push_back(&annotated_image);
}

void AnnotatedImages::write(FileStorage &fs)
{
	fs << "AnnotatedImages";
	fs << "{";
	fs << "name" << name << "ImagesAndObjects"
	   << "[";
	for (int index = 0; index < annotated_images.size(); index++)
		annotated_images[index]->write(fs);
	fs << "]"
	   << "}";
}
void AnnotatedImages::read(FileStorage &fs)
{
	FileNode node = fs.getFirstTopLevelNode();
	read(node);
}
void AnnotatedImages::read(FileNode &node)
{
	name = (string)node["name"];
	FileNode images_node = node["ImagesAndObjects"];
	if (images_node.type() == FileNode::SEQ)
	{
		for (FileNodeIterator it = images_node.begin(); it != images_node.end(); ++it)
		{
			FileNode current_node = *it;
			ImageWithBlueSignObjects *new_image_with_objects = new ImageWithBlueSignObjects(current_node);
			annotated_images.push_back(new_image_with_objects);
		}
	}
}
void AnnotatedImages::read(string filename)
{
	ImageWithBlueSignObjects *new_image_with_objects = new ImageWithBlueSignObjects(filename);
	annotated_images.push_back(new_image_with_objects);
}
void AnnotatedImages::LocateAndAddAllObjects(AnnotatedImages &training_images)
{
	for (int index = 0; index < annotated_images.size(); index++)
	{
		annotated_images[index]->LocateAndAddAllObjects(training_images);
	}
}
void AnnotatedImages::FindBestMatch(ObjectAndLocation *new_object) //Mat& perspective_warped_image, string& object_name, double& match_value)
{
	double match_value = -1.0;
	string object_name = "Unknown";
	double temp_best_match = 1000000.0;
	string temp_best_name;
	double temp_second_best_match = 1000000.0;
	string temp_second_best_name;
	for (int index = 0; index < annotated_images.size(); index++)
	{
		annotated_images[index]->FindBestMatch(new_object, object_name, match_value);
		if (match_value < temp_best_match)
		{
			if (temp_best_name.compare(object_name) != 0)
			{
				temp_second_best_match = temp_best_match;
				temp_second_best_name = temp_best_name;
			}
			temp_best_match = match_value;
			temp_best_name = object_name;
		}
		else if ((match_value != temp_best_match) && (match_value < temp_second_best_match) && (temp_best_name.compare(object_name) != 0))
		{
			temp_second_best_match = match_value;
			temp_second_best_name = object_name;
		}
	}
	if (temp_second_best_match / temp_best_match < REQUIRED_RATIO_OF_BEST_TO_SECOND_BEST)
		new_object->setName("Unknown");
	else
		new_object->setName(temp_best_name);
}

Mat AnnotatedImages::getImageOfAllObjects(int break_after)
{
	Mat all_rows_so_far;
	Mat output;
	int count = 0;
	int object_index = 0;
	string blank("");
	for (int index = 0; (index < annotated_images.size()); index++)
	{
		ObjectAndLocation *current_object = NULL;
		int object_index = 0;
		while ((current_object = (annotated_images[index])->getObject(object_index)) != NULL)
		{
			if (count == 0)
			{
				output = JoinSingleImage(current_object->getImage(), current_object->getName());
			}
			else if (count % break_after == 0)
			{
				if (count == break_after)
					all_rows_so_far = output;
				else
				{
					Mat temp_rows = JoinImagesVertically(all_rows_so_far, blank, output, blank, 0);
					all_rows_so_far = temp_rows.clone();
				}
				output = JoinSingleImage(current_object->getImage(), current_object->getName());
			}
			else
			{
				Mat new_output = JoinImagesHorizontally(output, blank, current_object->getImage(), current_object->getName(), 0);
				output = new_output.clone();
			}
			count++;
			object_index++;
		}
	}
	if (count == 0)
	{
		Mat blank_output(1, 1, CV_8UC3, Scalar(0, 0, 0));
		return blank_output;
	}
	else if (count < break_after)
		return output;
	else
	{
		Mat temp_rows = JoinImagesVertically(all_rows_so_far, blank, output, blank, 0);
		all_rows_so_far = temp_rows.clone();
		return all_rows_so_far;
	}
}

ImageWithObjects *AnnotatedImages::getAnnotatedImage(int index)
{
	if ((index >= 0) && (index < annotated_images.size()))
		return annotated_images[index];
	else
		return NULL;
}

ImageWithObjects *AnnotatedImages::FindAnnotatedImage(string filename_to_find)
{
	for (int index = 0; (index < annotated_images.size()); index++)
	{
		if (filename_to_find.compare(annotated_images[index]->filename) == 0)
			return annotated_images[index];
	}
	return NULL;
}

void MyApplication()
{
	AnnotatedImages trainingImages;
	FileStorage training_file("BlueSignsTraining.xml", FileStorage::READ);
	if (!training_file.isOpened())
	{
		cout << "Could not open the file: \""
			 << "BlueSignsTraining.xml"
			 << "\"" << endl;
	}
	else
	{
		trainingImages.read(training_file);
	}
	training_file.release();
	Mat image_of_all_training_objects = trainingImages.getImageOfAllObjects();
	imshow("All Training Objects", image_of_all_training_objects);
	imwrite("AllTrainingObjectImages.jpg", image_of_all_training_objects);
	char ch = cv::waitKey(1);

	AnnotatedImages groundTruthImages;
	FileStorage ground_truth_file("BlueSignsGroundTruth.xml", FileStorage::READ);
	if (!ground_truth_file.isOpened())
	{
		cout << "Could not open the file: \""
			 << "BlueSignsGroundTruth.xml"
			 << "\"" << endl;
	}
	else
	{
		groundTruthImages.read(ground_truth_file);
	}
	ground_truth_file.release();
	Mat image_of_all_ground_truth_objects = groundTruthImages.getImageOfAllObjects();
	imshow("All Ground Truth Objects", image_of_all_ground_truth_objects);
	imwrite("AllGroundTruthObjectImages.jpg", image_of_all_ground_truth_objects);
	ch = cv::waitKey(1);

	AnnotatedImages unknownImages("Blue Signs/Testing");
	unknownImages.LocateAndAddAllObjects(trainingImages);
	FileStorage unknowns_file("BlueSignsTesting.xml", FileStorage::WRITE);
	if (!unknowns_file.isOpened())
	{
		cout << "Could not open the file: \""
			 << "BlueSignsTesting.xml"
			 << "\"" << endl;
	}
	else
	{
		unknownImages.write(unknowns_file);
	}
	unknowns_file.release();
	Mat image_of_recognised_objects = unknownImages.getImageOfAllObjects();
	imshow("All Recognised Objects", image_of_recognised_objects);
	imwrite("AllRecognisedObjects.jpg", image_of_recognised_objects);

	ConfusionMatrix results(trainingImages);
	unknownImages.CompareObjectsWithGroundTruth(trainingImages, groundTruthImages, results);
	results.Print();

	Mat images_mat = trainingImages.getImageOfAllObjects();
	imshow("training_images", images_mat);
	waitKey(0);
}

bool PointInPolygon(Point2i point, vector<Point2i> vertices)
{
	int i, j, nvert = vertices.size();
	bool inside = false;

	for (i = 0, j = nvert - 1; i < nvert; j = i++)
	{
		if ((vertices[i].x == point.x) && (vertices[i].y == point.y))
			return true;
		if (((vertices[i].y >= point.y) != (vertices[j].y >= point.y)) &&
			(point.x <= (vertices[j].x - vertices[i].x) * (point.y - vertices[i].y) / (vertices[j].y - vertices[i].y) + vertices[i].x))
			inside = !inside;
	}
	return inside;
}

bool ObjectAndLocation::OverlapsWith(ObjectAndLocation *other_object)
{
	double area = contourArea(vertices);
	double other_area = contourArea(other_object->vertices);
	double overlap_area = 0.0;
	int count_points_inside = 0;
	for (int index = 0; (index < vertices.size()); index++)
	{
		if (PointInPolygon(vertices[index], other_object->vertices))
			count_points_inside++;
	}
	int count_other_points_inside = 0;
	for (int index = 0; (index < other_object->vertices.size()); index++)
	{
		if (PointInPolygon(other_object->vertices[index], vertices))
			count_other_points_inside++;
	}
	if (count_points_inside == vertices.size())
		overlap_area = area;
	else if (count_other_points_inside == other_object->vertices.size())
		overlap_area = other_area;
	else if ((count_points_inside == 0) && (count_other_points_inside == 0))
		overlap_area = 0.0;
	else
	{ // There is a partial overlap of the polygons.
		// Find min & max x & y for the current object
		int min_x = vertices[0].x, min_y = vertices[0].y, max_x = vertices[0].x, max_y = vertices[0].y;
		for (int index = 0; (index < vertices.size()); index++)
		{
			if (min_x > vertices[index].x)
				min_x = vertices[index].x;
			else if (max_x < vertices[index].x)
				max_x = vertices[index].x;
			if (min_y > vertices[index].y)
				min_y = vertices[index].y;
			else if (max_y < vertices[index].y)
				max_y = vertices[index].y;
		}
		int min_x2 = other_object->vertices[0].x, min_y2 = other_object->vertices[0].y, max_x2 = other_object->vertices[0].x, max_y2 = other_object->vertices[0].y;
		for (int index = 0; (index < other_object->vertices.size()); index++)
		{
			if (min_x2 > other_object->vertices[index].x)
				min_x2 = other_object->vertices[index].x;
			else if (max_x2 < other_object->vertices[index].x)
				max_x2 = other_object->vertices[index].x;
			if (min_y2 > other_object->vertices[index].y)
				min_y2 = other_object->vertices[index].y;
			else if (max_y2 < other_object->vertices[index].y)
				max_y2 = other_object->vertices[index].y;
		}
		// We only need the maximum overlapping bounding boxes
		if (min_x < min_x2)
			min_x = min_x2;
		if (max_x > max_x2)
			max_x = max_x2;
		if (min_y < min_y2)
			min_y = min_y2;
		if (max_y > max_y2)
			max_y = max_y2;
		// For all points
		overlap_area = 0;
		Point2i current_point;
		// Try ever decreasing squares within the overlapping (image aligned) bounding boxes to find the overlapping area.
		bool all_points_inside = false;
		int distance_from_edge = 0;
		for (; ((distance_from_edge < (max_x - min_x + 1) / 2) && (distance_from_edge < (max_y - min_y + 1) / 2) && (!all_points_inside)); distance_from_edge++)
		{
			all_points_inside = true;
			for (current_point.x = min_x + distance_from_edge; (current_point.x <= (max_x - distance_from_edge)); current_point.x++)
				for (current_point.y = min_y + distance_from_edge; (current_point.y <= max_y - distance_from_edge); current_point.y += max_y - 2 * distance_from_edge - min_y)
				{
					if ((PointInPolygon(current_point, vertices)) && (PointInPolygon(current_point, other_object->vertices)))
						overlap_area++;
					else
						all_points_inside = false;
				}
			for (current_point.y = min_y + distance_from_edge + 1; (current_point.y <= (max_y - distance_from_edge - 1)); current_point.y++)
				for (current_point.x = min_x + distance_from_edge; (current_point.x <= max_x - distance_from_edge); current_point.x += max_x - 2 * distance_from_edge - min_x)
				{
					if ((PointInPolygon(current_point, vertices)) && (PointInPolygon(current_point, other_object->vertices)))
						overlap_area++;
					else
						all_points_inside = false;
				}
		}
		if (all_points_inside)
			overlap_area += (max_x - min_x + 1 - 2 * (distance_from_edge + 1)) * (max_y - min_y + 1 - 2 * (distance_from_edge + 1));
	}
	double percentage_overlap = (overlap_area * 2.0) / (area + other_area);
	return (percentage_overlap >= REQUIRED_OVERLAP);
}

void AnnotatedImages::CompareObjectsWithGroundTruth(AnnotatedImages &training_images, AnnotatedImages &ground_truth, ConfusionMatrix &results)
{
	// For every annotated image in ground_truth, find the corresponding image in this
	for (int ground_truth_image_index = 0; ground_truth_image_index < ground_truth.annotated_images.size(); ground_truth_image_index++)
	{
		ImageWithObjects *current_annotated_ground_truth_image = ground_truth.annotated_images[ground_truth_image_index];
		ImageWithObjects *current_annotated_recognition_image = FindAnnotatedImage(current_annotated_ground_truth_image->filename);
		if (current_annotated_recognition_image != NULL)
		{
			ObjectAndLocation *current_ground_truth_object = NULL;
			int ground_truth_object_index = 0;
			Mat *display_image = NULL;
			if (!current_annotated_recognition_image->image.empty())
			{
				display_image = &(current_annotated_recognition_image->image);
			}
			// For each object in ground_truth.annotated_image
			while ((current_ground_truth_object = current_annotated_ground_truth_image->getObject(ground_truth_object_index)) != NULL)
			{
				if ((current_ground_truth_object->getMinimumSideLength() >= MINIMUM_SIGN_SIDE) &&
					(current_ground_truth_object->getArea() >= MINIMUM_SIGN_AREA))
				{
					// Determine the number of overlapping objects (correct & incorrect)
					vector<ObjectAndLocation *> overlapping_correct_objects;
					vector<ObjectAndLocation *> overlapping_incorrect_objects;
					ObjectAndLocation *current_recognised_object = NULL;
					int recognised_object_index = 0;
					// For each object in this.annotated_image
					while ((current_recognised_object = current_annotated_recognition_image->getObject(recognised_object_index)) != NULL)
					{
						if (current_recognised_object->getName().compare("Unknown") != 0)
							if (current_ground_truth_object->OverlapsWith(current_recognised_object))
							{
								if (current_ground_truth_object->getName().compare(current_recognised_object->getName()) == 0)
									overlapping_correct_objects.push_back(current_recognised_object);
								else
									overlapping_incorrect_objects.push_back(current_recognised_object);
							}
						recognised_object_index++;
					}
					if ((overlapping_correct_objects.size() == 0) && (overlapping_incorrect_objects.size() == 0))
					{
						if (display_image != NULL)
						{
							Scalar colour(0x00, 0x00, 0xFF);
							current_ground_truth_object->DrawObject(display_image, colour);
						}
						results.AddFalseNegative(current_ground_truth_object->getName());
						cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (False Negative) , " << current_ground_truth_object->getVerticesString() << endl;
					}
					else
					{
						for (int index = 0; (index < overlapping_correct_objects.size()); index++)
						{
							Scalar colour(0x00, 0xFF, 0x00);
							results.AddMatch(current_ground_truth_object->getName(), overlapping_correct_objects[index]->getName(), (index > 0));
							if (index > 0)
							{
								colour[2] = 0xFF;
								cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (Duplicate) , " << current_ground_truth_object->getVerticesString() << endl;
							}
							if (display_image != NULL)
								current_ground_truth_object->DrawObject(display_image, colour);
						}
						for (int index = 0; (index < overlapping_incorrect_objects.size()); index++)
						{
							if (display_image != NULL)
							{
								Scalar colour(0xFF, 0x00, 0xFF);
								overlapping_incorrect_objects[index]->DrawObject(display_image, colour);
							}
							results.AddMatch(current_ground_truth_object->getName(), overlapping_incorrect_objects[index]->getName(), (index > 0));
							cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (Mismatch), " << overlapping_incorrect_objects[index]->getName() << " , " << current_ground_truth_object->getVerticesString() << endl;
							;
						}
					}
				}
				else
					cout << current_annotated_ground_truth_image->filename << ", " << current_ground_truth_object->getName() << ", (DROPPED GT) , " << current_ground_truth_object->getVerticesString() << endl;

				ground_truth_object_index++;
			}
			//	For each object in this.annotated_image
			//				For each overlapping object in ground_truth.annotated_image
			//					Don't do anything (as already done above)
			//			If no overlapping objects.
			//				Update the confusion table (with a False Positive)
			ObjectAndLocation *current_recognised_object = NULL;
			int recognised_object_index = 0;
			// For each object in this.annotated_image
			while ((current_recognised_object = current_annotated_recognition_image->getObject(recognised_object_index)) != NULL)
			{
				if ((current_recognised_object->getMinimumSideLength() >= MINIMUM_SIGN_SIDE) &&
					(current_recognised_object->getArea() >= MINIMUM_SIGN_AREA))
				{
					// Determine the number of overlapping objects (correct & incorrect)
					vector<ObjectAndLocation *> overlapping_objects;
					ObjectAndLocation *current_ground_truth_object = NULL;
					int ground_truth_object_index = 0;
					// For each object in ground_truth.annotated_image
					while ((current_ground_truth_object = current_annotated_ground_truth_image->getObject(ground_truth_object_index)) != NULL)
					{
						if (current_ground_truth_object->OverlapsWith(current_recognised_object))
							overlapping_objects.push_back(current_ground_truth_object);
						ground_truth_object_index++;
					}
					if ((overlapping_objects.size() == 0) && (current_recognised_object->getName().compare("Unknown") != 0))
					{
						results.AddFalsePositive(current_recognised_object->getName());
						if (display_image != NULL)
						{
							Scalar colour(0x7F, 0x7F, 0xFF);
							current_recognised_object->DrawObject(display_image, colour);
						}
						cout << current_annotated_recognition_image->filename << ", " << current_recognised_object->getName() << ", (False Positive) , " << current_recognised_object->getVerticesString() << endl;
					}
				}
				else
					cout << current_annotated_recognition_image->filename << ", " << current_recognised_object->getName() << ", (DROPPED) , " << current_recognised_object->getVerticesString() << endl;
				recognised_object_index++;
			}
			if (display_image != NULL)
			{
				Mat smaller_image;
				resize(*display_image, smaller_image, Size(display_image->cols / 4, display_image->rows / 4));
				imshow(current_annotated_recognition_image->filename, smaller_image);
				char ch = cv::waitKey(1);
				//				delete display_image;
			}
		}
	}
}

// Determine object classes from the training_images (vector of strings)
// Create and zero a confusion matrix
ConfusionMatrix::ConfusionMatrix(AnnotatedImages training_images)
{
	// Extract object class names
	ImageWithObjects *current_annnotated_image = NULL;
	int image_index = 0;
	while ((current_annnotated_image = training_images.getAnnotatedImage(image_index)) != NULL)
	{
		ObjectAndLocation *current_object = NULL;
		int object_index = 0;
		while ((current_object = current_annnotated_image->getObject(object_index)) != NULL)
		{
			AddObjectClass(current_object->getName());
			object_index++;
		}
		image_index++;
	}
	// Create and initialise confusion matrix
	confusion_size = class_names.size() + 1;
	confusion_matrix = new int *[confusion_size];
	for (int index = 0; (index < confusion_size); index++)
	{
		confusion_matrix[index] = new int[confusion_size];
		for (int index2 = 0; (index2 < confusion_size); index2++)
			confusion_matrix[index][index2] = 0;
	}
	false_index = confusion_size - 1;
}
void ConfusionMatrix::AddObjectClass(string object_class_name)
{
	int index = getObjectClassIndex(object_class_name);
	if (index == -1)
		class_names.push_back(object_class_name);
	tp = fp = fn = 0;
}
int ConfusionMatrix::getObjectClassIndex(string object_class_name)
{
	int index = 0;
	for (; (index < class_names.size()) && (object_class_name.compare(class_names[index]) != 0); index++)
		;
	if (index < class_names.size())
		return index;
	else
		return -1;
}
void ConfusionMatrix::AddMatch(string ground_truth, string recognised_as, bool duplicate)
{
	if ((ground_truth.compare(recognised_as) == 0) && (duplicate))
		AddFalsePositive(recognised_as);
	else
	{
		confusion_matrix[getObjectClassIndex(ground_truth)][getObjectClassIndex(recognised_as)]++;
		if (ground_truth.compare(recognised_as) == 0)
			tp++;
		else
		{
			fp++;
			fn++;
		}
	}
}
void ConfusionMatrix::AddFalseNegative(string ground_truth)
{
	fn++;
	confusion_matrix[getObjectClassIndex(ground_truth)][false_index]++;
}
void ConfusionMatrix::AddFalsePositive(string recognised_as)
{
	fp++;
	confusion_matrix[false_index][getObjectClassIndex(recognised_as)]++;
}
void ConfusionMatrix::Print()
{
	cout << ",,,Recognised as:" << endl
		 << ",,";
	for (int recognised_as_index = 0; recognised_as_index < confusion_size; recognised_as_index++)
		if (recognised_as_index < confusion_size - 1)
			cout << class_names[recognised_as_index] << ",";
		else
			cout << "False Negative,";
	cout << endl;
	for (int ground_truth_index = 0; (ground_truth_index <= class_names.size()); ground_truth_index++)
	{
		if (ground_truth_index < confusion_size - 1)
			cout << "Ground Truth," << class_names[ground_truth_index] << ",";
		else
			cout << "Ground Truth,False Positive,";
		for (int recognised_as_index = 0; recognised_as_index < confusion_size; recognised_as_index++)
			cout << confusion_matrix[ground_truth_index][recognised_as_index] << ",";
		cout << endl;
	}
	double precision = ((double)tp) / ((double)(tp + fp));
	double recall = ((double)tp) / ((double)(tp + fn));
	double f1 = 2.0 * precision * recall / (precision + recall);
	cout << endl
		 << "Precision = " << precision << endl
		 << "Recall = " << recall << endl
		 << "F1 = " << f1 << endl;
}

void ObjectAndLocation::setImage(Mat object_image)
{
	image = object_image.clone();
	// *** Student should add any initialisation (of their images or features; see private data below) they wish into this method.
}

void ImageWithBlueSignObjects::AddSignObject(Mat img, Rect crop)
{
	// Fix the croping of blue signs with new thresholding.
	


	Mat sign = img(crop);
	Mat img_out = img.clone();
	Mat colourSign = img(crop);

	// Scalar colour(0x00, 0x00, 0xff);
	// copyMakeBorder(colourSign, colourSign, 10, 10, 10, 10, BORDER_CONSTANT, colour);

	// vector<Mat> channels(3);
	// split(colourSign, channels);
	// img = channels[2];
	// threshold(img, img, 245, 255, THRESH_BINARY_INV | THRESH_OTSU);

	// int morph_size = 2;
	// Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	// morphologyEx(img, img, MORPH_OPEN, element, Point(-1, -1), 2);

	Canny(sign, img, 50, 255, 3);

	//using contours define a aproxpoloy to descript sign outline
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<vector<Point2f>> approxPoly(contours.size());

	double max_area = 0;
	int index = 0;

	for (size_t i = 0; i < contours.size(); i++)
	{
		// if ((hierarchy[i][2] != -1) && (hierarchy[i][3] == -1))
		// {
			double epsilon = .08 * arcLength(contours[i], true);
			approxPolyDP(contours[i], approxPoly[i], epsilon, true);
			double area = contourArea(contours[i]);
			if(area > max_area){
				max_area = area;
				index = i;
			}
		// }
	}
	cout << max_area << " @ " << index << endl;

	//using the apromiation of the bounds transform the image to a 100 x 100 square
	float imageWidth = 100.0f;
	float imageHeight = 100.0f;
	Point2f dst_points[4] = {Point2f(1.0f, 1.0f), Point2f(imageWidth, 1.0f), Point2f(1.0f, imageHeight), Point2f(imageWidth, imageHeight)};
	Point2f rect_points[4];



	// // find contour of larget area
	// for (size_t i = 0; i < contours.size(); i++)
	// {
	// 	if ((hierarchy[i][2] != -1) && (hierarchy[i][3] == -1))
	// 	{
		if(max_area > 0){
			for (int j = 0; j < 4; j++)
			{
				rect_points[j] = approxPoly[index][j];
			}
		}
	// 	}
	// }

	//determine which corners are which using their reltive distance from orignal crop
	float minSum = FLT_MAX;
	float maxSum = FLT_MIN;
	float minDiff = FLT_MAX;
	float maxDiff = FLT_MIN;
	Point2i topLeft, topRight, botLeft, botRight = Point2f(-1, -1);
	for (int i = 0; i < 4; i++)
	{
		float sum = rect_points[i].x + rect_points[i].y;
		float diff = rect_points[i].x - rect_points[i].y;

		if (sum <= minSum)
		{
			minSum = sum;
			topLeft = (Point2i)rect_points[i];
		}
		if (diff >= maxDiff)
		{
			maxDiff = diff;
			topRight = (Point2i)rect_points[i];
		}
		if (diff <= minDiff)
		{
			minDiff = diff;
			botLeft = (Point2i)rect_points[i];
		}
		if (sum >= maxSum)
		{
			maxSum = sum;
			botRight = (Point2i)rect_points[i];
		}
	}

	//take found corners and relate them to their position in original image
	vector<Point2i> corners;
	corners.push_back((Point2i)(crop.tl() + topLeft));
	corners.push_back((Point2i)(crop.tl() + topRight));
	corners.push_back((Point2i)(crop.tl() + botRight));
	corners.push_back((Point2i)(crop.tl() + botLeft));

	//space around image
	float offset = 0.1 * STANDARD_SIGN_WIDTH_AND_HEIGHT;

	//geometrically transform image
	Mat perspective_warped_image = Mat::zeros(STANDARD_SIGN_WIDTH_AND_HEIGHT, STANDARD_SIGN_WIDTH_AND_HEIGHT, image.type());
	Mat perspective_matrix(3, 3, CV_32FC1);
	Point2f source_points[4] = {{float(corners[0].x), float(corners[0].y)}, {float(corners[1].x), float(corners[1].y)}, {float(corners[2].x), float(corners[2].y)}, {float(corners[3].x), float(corners[3].y)}};
	Point2f destination_points[4] = {{offset, offset}, {STANDARD_SIGN_WIDTH_AND_HEIGHT - offset, offset}, {STANDARD_SIGN_WIDTH_AND_HEIGHT - offset, STANDARD_SIGN_WIDTH_AND_HEIGHT - offset}, {offset, STANDARD_SIGN_WIDTH_AND_HEIGHT - offset}};
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);
	warpPerspective(img_out, perspective_warped_image, perspective_matrix, perspective_warped_image.size());

	//add object if the contour are is close to the area of the bounding box
	if (contourArea(corners) / crop.area() > .8)
	{
		addObject("", corners[0].x, corners[0].y, corners[1].x, corners[1].y,
				  corners[2].x, corners[2].y, corners[3].x, corners[3].y, perspective_warped_image);
	}
}

void ImageWithBlueSignObjects::LocateAndAddAllObjects(AnnotatedImages &training_images)
{
	cout << "Detecting Objects for " << filename << endl;
	Mat img, colourMat, output;
	output = image.clone();

	//original: Red colour channel, brightness -50, threshold 60
	//analyse the red channel and threshold
	image.convertTo(colourMat, -1, 1, -50);
	vector<Mat> channels(3);
	split(colourMat, channels);
	img = channels[2];
	bitwise_not(img, img);
	// threshold(img, img, 200, 255,THRESH_BINARY);
	adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 75, 2);

	// Morpholody to clean the image for edge detection
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(img, img, MORPH_OPEN, element, Point(-1, -1), 3);

	//find egdes
	Canny(img, img, 150, 255, 3);

	//find contors on edge image
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	//find bounding retangles of countors
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}

	//crop image based off coverage of bounding box and hierachy of contour
	for (size_t i = 0; i < contours.size(); i++)
	{
		double coverage, area;
		area = boundRect[i].area();
		coverage = ((contourArea(contours[i])) / area);

		if ((coverage > 0.6) && (area > 5000) && (hierarchy[i][2] != -1) && (hierarchy[i][3] == -1))
		{
			Rect crop = boundRect[i];
			AddSignObject(output, crop);
		}
	}

	//Reconise each found object against training images, set the name of the object based off the best match
	for (size_t i = 0; i < objects.size(); i++)
	{
		double min_val = 1;
		string bestMatch = "Unknown";
		vector<Point2f> features;
		objects[i].FeatureVector(features);
		objects[i].setFeatures(features);

		vector<ImageWithObjects *> imgWithObj = training_images.annotated_images;

		//for each training image compare
		for (size_t j = 0; j < imgWithObj.size(); j++)
		{
			ImageWithObjects *img = imgWithObj[j];

			double matching_value = 1;
			string name = objects[i].getName();
			ObjectAndLocation *obj = &objects[i];

			(*img).FindBestMatch(obj, name, matching_value);
			if (matching_value < min_val)
			{
				min_val = matching_value;
				bestMatch = name;
			}
		}
		objects[i].setName(bestMatch);
		cout << "Object found: " << bestMatch << endl;
	}
}

vector<Point2f> ObjectAndLocation::getFeatures()
{
	return featureVector;
}

void ObjectAndLocation::setFeatures(vector<Point2f> features)
{
	featureVector = features;
}

void ObjectAndLocation::FeatureVector(vector<Point2f> &features)
{
	Mat img;
	vector<Mat> channels(3);
	split(image, channels);
	img = channels[2];
	bitwise_not(img, img);
	threshold(img, img, 100, 255, THRESH_BINARY);

	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(img, img, MORPH_OPEN, element, Point(-1, -1), 3);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<vector<Point2f>> approxPoly(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		double epsilon = .01 * arcLength(contours[i], true);
		approxPolyDP(contours[i], approxPoly[i], epsilon, true);
	}

	vector<Point2f> foundFeatures;
	double count = 0.0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] != -1)
		{
			double area = contourArea(approxPoly[i]);
			foundFeatures.push_back(Point2d((double)i, area));
		}
	}

	features = foundFeatures;
}

#define BAD_MATCHING_VALUE 1000000000.0;
double ObjectAndLocation::compareObjects(ObjectAndLocation *otherObject)
{
	ObjectAndLocation obj = *otherObject;
	Mat img = obj.image.clone();
	Mat templ = image.clone();

	Mat matching_space;

	//the image has a sign area of around .90 thus scale template and search for best match
	double factor = 0.80;
	double maxFactor = 1.00;
	double threshold = 0.9;
	double lowestVal = threshold;
	while (factor <= maxFactor)
	{
		Mat scale_template;
		//resize image based off factor
		resize(templ, scale_template, Size(0, 0), factor, factor, INTER_AREA);
		matching_space.create(img.cols - scale_template.cols + 1, img.rows - scale_template.cols + 1, CV_32FC1);
		matchTemplate(img, scale_template, matching_space, CV_TM_SQDIFF_NORMED);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;

		minMaxLoc(matching_space, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		//set the lowestVal if minval is lower and interate scaling factor
		if (minVal < lowestVal)
			lowestVal = minVal;
		factor += 0.01;
	}
	if (lowestVal >= threshold)
		return BAD_MATCHING_VALUE;
	return lowestVal;
}

int main(int argc, char **argv)
{
	MyApplication();
	return 0;
}