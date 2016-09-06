#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

// все растры должны быть одного размера, иначе задача не будет иметь смысла
bool scenesSameSizeTest(const std::vector<cv::Mat> &scenes)
{
	bool bTheSame = true;
	assert(scenes.size() > 1);

	int width = scenes[0].cols, height = scenes[0].rows;

	for (auto const& img : scenes) {
		if (img.cols != width || img.rows != height) { bTheSame = false; break; }
	}

	return bTheSame;
}

// оверлей (with alpha)
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location)
{
	background.copyTo(output);

	for (int y = std::max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y;
		if (fY >= foreground.rows) { break; }
		for (int x = std::max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= foreground.cols) { break; }

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.0;

			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y * output.step + output.channels() * x + c] =
					cvRound(backgroundPx * (1.0 - opacity) + foregroundPx * opacity);
			}
		}
	}
}

/*hsv2rgb snippet*/
cv::Scalar hsv2bgr(double h, double s, double v) // angle, percent, percent
{
	double      hh, p, q, t, ff;
	long        i;
	double		r_o, g_o, b_o;

	if (s <= 0.0) {
		r_o = v;
		g_o = v;
		b_o = v;
		return cv::Scalar(b_o * 255, g_o*255, r_o * 255);
	}
	hh = h;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = v * (1.0 - s);
	q = v * (1.0 - (s * ff));
	t = v * (1.0 - (s * (1.0 - ff)));

	switch (i) {
	case 0:
		r_o = v;
		g_o = t;
		b_o = p;
		break;
	case 1:
		r_o = q;
		g_o = v;
		b_o = p;
		break;
	case 2:
		r_o = p;
		g_o = v;
		b_o = t;
		break;
	case 3:
		r_o = p;
		g_o = q;
		b_o = v;
		break;
	case 4:
		r_o = t;
		g_o = p;
		b_o = v;
		break;
	case 5:
	default:
		r_o = v;
		g_o = p;
		b_o = q;
		break;
	}

	return cv::Scalar(b_o * 255, g_o * 255, r_o * 255);
}

// Получить цвет фона можно разным путем, т.е. он однородный, после выделения с порогом можно взять любую точку вне контура.
// Можно к-measure (2) [метод к-средний] кластеризации,
// А можно попробовать просто проанализировать с определенной точностью hsv-гистограмму, выбрать макс. значение -> hsv -> bgr
// чтобы не нагружать память, используем часть раскадровки, a лучше пусть будет первая раскадровка.
cv::Scalar GetDominantBGR(const std::vector<cv::Mat>& scenes)
{
	cv::Mat image_hsv;

	cv::cvtColor(scenes[0], image_hsv, CV_BGR2HSV); // take 1st frame

	// Quanta Ratio
	int scale = 10;

	int hbins = 36, sbins = 25, vbins = 25;
	int histSize[] = { hbins, sbins, vbins };

	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	float vranges[] = { 0, 256 };

	const float* ranges[] = { hranges, sranges, vranges };
	cv::MatND hist;

	int channels[] = { 0, 1, 2 };

	cv::calcHist(&image_hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges);

	int maxVal = 0;
	int hue = 0, saturation = 0, value = 0;

	for (int h = 0; h < hbins; h++)
		for (int s = 0; s < sbins; s++)
			for (int v = 0; v < vbins; v++)
			{
				int binVal = hist.at<int>(h, s, v);
				if (binVal > maxVal)
				{
					maxVal = binVal;

					hue = h;
					saturation = s;
					value = v;
				}
			}

	hue = hue * scale; // angle 0..360
	saturation = saturation * scale; // 0..255
	value = value * scale; // 0..255

	return hsv2bgr(hue /*0..360 degrees*/, ((double)saturation)/255 /*percentes: 0..1*/, ((double)value)/255 /*percentes: 0..1*/);
}

// преобразование координат линии (по заданию) в координаты opencv
std::pair<cv::Point, cv::Point> projCoord(const std::pair<cv::Point2d, cv::Point2d>& line, const std::vector<cv::Mat>& scenes)
{
	int x1 = cvRound(line.first.x * scenes[0].cols);
	int x2 = cvRound(line.second.x * scenes[0].cols);

	int y1 = cvRound((1.0 - line.first.y) * scenes[0].rows);
	int y2 = cvRound((1.0 - line.second.y) * scenes[0].rows);

	return std::pair<cv::Point, cv::Point>(cv::Point(x1, y1), cv::Point(x2, y2));
}

// Определяем пересечение линий (сегментов)
bool line_intersects(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, cv::Point* out = NULL)
{
	int s1_x, s1_y, s2_x, s2_y;
	s1_x = p1.x - p0.x; s1_y = p1.y - p0.y;
	s2_x = p3.x - p2.x; s2_y = p3.y - p2.y;

	float s, t;
	int d = -s2_x * s1_y + s1_x * s2_y;
	if (!d) return false;

	s = (float)(-s1_y * (p0.x - p2.x) + s1_x * (p0.y - p2.y)) / d;
	t = (float)(s2_x * (p0.y - p2.y) - s2_y * (p0.x - p2.x)) / d;

	if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
	{
		if (out != NULL)
		{
			*out = cv::Point(cvRound(p0.x + (t * s1_x)), cvRound(p0.y + (t * s1_y)));
		}

		return true;
	}

	return false;
}

//Определяем пересечение линии и rect
// LEFT = 2, BOTTOM = 4, RIGHT = 8, TOP = 16
bool line_rect_inersect(const cv::Point& p0, const cv::Point& p1, const cv::Rect& bbox, int *num = 0)
{
	bool result = false;
	if (num) *num = 0;
	if (line_intersects(p0, p1, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.width, bbox.y))) { result = true; if (num) (*num) &= 16; }
	if (line_intersects(p0, p1, cv::Point(bbox.x + bbox.width, bbox.y), cv::Point(bbox.x + bbox.width, bbox.y + bbox.height))) { result = true; if (num) (*num) &= 8; }
	if (line_intersects(p0, p1, cv::Point(bbox.x, bbox.y + bbox.height), cv::Point(bbox.x + bbox.width, bbox.y + bbox.height))) { result = true; if (num) (*num) &= 4; }
	if (line_intersects(p0, p1, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x, bbox.y + bbox.height))) { result = true; if (num) (*num) &= 2; }
	return result;
}


// внутр. угол между 2 линиями
double angleBetween2Lines(cv::Point line1Start, cv::Point line1End, cv::Point line2Start, cv::Point line2End) {
	double x1 = line1Start.x - line1End.x;
	double y1 = line1Start.y - line1End.y;
	double x2 = line2Start.x - line2End.x;
	double y2 = line2Start.y - line2End.y;

	double angle1, angle2, angle;

	if (x1 != 0.0f)
		angle1 = atan(y1 / x1);
	else
		angle1 = CV_PI / 2.0;	// 90 degrees

	if (x2 != 0.0f)
		angle2 = atan(y2 / x2);
	else
		angle2 = CV_PI / 2.0;	// 90 degrees
								//
	angle = fabs(angle2 - angle1);
	angle = angle * 180.0 / CV_PI;
	return angle;
}

// все точки пересечения с контуром
std::vector<cv::Point> getIntersectionPoints(const std::vector<cv::Point>& contour, const cv::Point& lineStart, const cv::Point& lineEnd) {
	std::vector<cv::Point> result;
	for (int i = 0; i < contour.size(); ++i) {
		cv::Point i_p, nxt_p;
		if (i == contour.size() - 1) {
			nxt_p = contour[0];
		}
		else
		{
			nxt_p = contour[i + 1];
		}

		if (line_intersects(lineStart, lineEnd, contour[i], nxt_p, &i_p)) {
			result.push_back(i_p);
		}
	}

	return result;
}

// поиск сложного сечения
double findSection(const cv::Scalar& bg, const cv::Mat& obj, const cv::Point& loc, 
						const cv::Point& lineStart, const cv::Point& lineEnd, const std::vector<cv::Mat>& scenes)
{
	cv::Mat background(scenes[0].size(), scenes[0].type(), bg), frame;
	overlayImage(background, obj, frame, loc);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::cvtColor(frame, frame, CV_BGR2GRAY);
	cv::threshold(frame, frame, 100, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
	cv::findContours(frame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	double result = .0;

	if (!contours.empty() && !hierarchy.empty())
	{
		//max
		double largest_area = 0;
		int largest_contour_index = 0;
		for (int i = 0; i < contours.size(); ++i)
		{
			double area = contourArea(contours[i], false);
			if (area > largest_area) {
				largest_area = area;
				largest_contour_index = i;
			}
		}

		if (line_rect_inersect(lineStart, lineEnd, cv::boundingRect(contours[largest_contour_index]))) {
			for (int i = 0; i<contours.size(); ++i) {
				auto i_p = getIntersectionPoints(contours[i], lineStart, lineEnd);
				if (i_p.size())
				{
					bool lineEndInTheContour = (i_p.size() % 2 != 0);
					double dist = .0;

					if (hierarchy[i][3] != -1) { // hole
						for (int j = 0; j < i_p.size() - 1; ++j) {
							dist -= cv::norm(i_p[j + 1] - i_p[j]);
						}
					}
					else {
						for (int j = 0; j < i_p.size() - 1; ++j) {
							dist += cv::norm(i_p[j + 1] - i_p[j]);
						}
					}

					if (lineEndInTheContour) { // probably
						double minDist = .0;
						cv::Point currPoint;
						if (cv::pointPolygonTest(contours[i], lineStart, false) > 0) {
							//lineStart
							if (cv::norm(i_p[0] - lineStart) < cv::norm(i_p[i_p.size() - 1] - lineStart)) {
								minDist = cv::norm(i_p[0] - lineStart);
							}
							else {
								minDist = cv::norm(i_p[i_p.size() - 1] - lineStart);
							}
						}
						else if (cv::pointPolygonTest(contours[i], lineEnd, false) > 0) {
							// lineEnd
							if (cv::norm(i_p[0] - lineEnd) < cv::norm(i_p[i_p.size() - 1] - lineEnd)) {
								minDist = cv::norm(i_p[0] - lineEnd);
							}
							else {
								minDist = cv::norm(i_p[i_p.size() - 1] - lineEnd);
							}
						}

						if (hierarchy[i][3] != -1) { // hole
							dist -= minDist;
						}
						else {
							dist += minDist;
						}
					}

					result += dist;
				}
			}
		}
		else {
			return 0;
		}
	}

	return result;
}

int main(int argc, const char* argv[])
{
	setlocale(LC_ALL, "Russian");

	if (argc != 6) {
		std::cerr << "Неверное колличество параметров. Пример: prog.exe \"fullpath_wildcard\" line_x1 line_y1 line_x2 line_y2, где line_* = [0..1]." << std::endl;
		system("pause");
		return -1;
	}

	std::string path = argv[1];

	double lineCoordinates[4];
	for (int i = 0; i < 4; ++i) {
		std::string input = argv[i + 2];
		std::stringstream(input) >> lineCoordinates[i];
	}

	//загружаем scenes
	std::vector<cv::String> fn;
	
	cv::glob(path, fn, false);

	if (fn.empty())
	{
		std::cerr << "По данному пути с wildcard не найденно подходящих раскадровок." << std::endl;
		system("pause");
		return -1;
	}

	if (fn.size() < 2)
	{
		std::cerr << "Раскадровок должно быть больше двух." << std::endl;
		system("pause");
		return -1;
	}

	for (int i = 0; i < 4; ++i) {
		if (lineCoordinates[i] < .0 || lineCoordinates[i] > 1.0)
		{
			std::cerr << i + 1 <<"-ая координата линии имеет неверный формат [0.0 .. 1.0]" << std::endl;
			system("pause");
			return -1;
		}
	}

	std::sort(fn.begin(), fn.end(), [](const cv::String& a, const cv::String& b) {return a < b; }); // по заданию, в папке упорядоченный набор кадров (ASC)
	
	std::vector<cv::Mat> scenes, scenes_grey;
	std::for_each(fn.begin(), fn.end(), [&scenes, &scenes_grey](const cv::String& fn_img) { 
		scenes.push_back(cv::imread(fn_img, CV_LOAD_IMAGE_COLOR)); 
		scenes_grey.push_back(cv::imread(fn_img, CV_LOAD_IMAGE_GRAYSCALE)); 
	});

	if (!scenesSameSizeTest(scenes)) {
		std::cerr << "Все раскадровки должны быть одинакового размера, иначе задача не имеет смысла." << std::endl;
		system("pause");
		return -1;
	}

	// линия по заданию (x=0..1, y =0..1, точка (0,0) – левый нижний угол, точка (1, 1) – правый верхний угол кадра
	std::pair<cv::Point2d, cv::Point2d> input_line =
			{ cv::Point2d(lineCoordinates[0],lineCoordinates[1]), cv::Point2d(lineCoordinates[2],lineCoordinates[3]) };
	auto user_line = projCoord(input_line, scenes);

	// Находим объект
	// findContours принимает одноканальные изображения, так что конвертируем наши растры из 3-х каналов BGR
	// не забываем что контур может быть сложным, но одним, используем hierarchy.
	// находим максимальный S - будет контур, который поместился весь в кадре всей раскадровки. (иначе в раскадровке не было целого объекта никогда)
	// не забываем что объект должен быть один и может выходить за границы раскадровки
	// выводим наш контур на новое изображение с альфа каналом
	// не забываем что объект может содержать несколько цветов и быть любой формы,
	// в этом случае findContours может вернуть много контуров, надо выявить максимальный. И с каждым кадром работать только с максимальным контуром.

	double g_area = 0;
	int g_scene_i = 0;
	std::vector<std::vector<cv::Point>> g_contours;
	std::vector<cv::Vec4i> g_hierarchy;
	int g_largest_contour_index = 0;
	int g_i = 0;

	for (auto const& img : scenes_grey) {
		cv::Mat tmp;
		int largest_contour_index = 0;
		double largest_area = 0;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::threshold(img, tmp, 100, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU); // средний порог // adaptive for interesting tests?

		cv::findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < contours.size(); ++i)
		{
			double area = contourArea(contours[i], false);
			if (area > largest_area) {
				largest_area = area;
				largest_contour_index = i;
			}
		}

		if (largest_area > g_area)
		{
			g_area = largest_area;
			g_scene_i = g_i;
			g_contours.clear();
			g_hierarchy.clear();
			g_contours = contours;
			g_hierarchy = hierarchy;
			g_largest_contour_index = largest_contour_index;
		}

		++g_i;
	}

	// на данном этапе у нас есть полноценный контур со средним порогом
	cv::Mat alpha(scenes[g_scene_i].size(), CV_8UC1, cv::Scalar(0));
	cv::Rect R = cv::boundingRect(g_contours[g_largest_contour_index]);
	
	cv::drawContours(alpha, g_contours, g_largest_contour_index, cv::Scalar(255), CV_FILLED, 8, g_hierarchy);

	cv::Mat rgb[3];
	cv::split(scenes[g_scene_i], rgb);

	cv::Mat rgba[4] = {rgb[0], rgb[1], rgb[2], alpha};
	cv::Mat dst;
	cv::merge(rgba, 4, dst);

	//определяем цвет фона (а применим-ка тут метод поиска доминантного цвета в кадре раскадровки)
	cv::Scalar background = GetDominantBGR(scenes);

	//делаем раскадровку 4-х канальной (+ альфа-канал)
	std::for_each(scenes.begin(), scenes.end(), [](cv::Mat& scene) { cv::cvtColor(scene, scene, CV_BGR2BGRA); });

	// получили 4-х канальное изображение объекта в obj
	// 4 channel obj
	cv::Mat object = dst(cv::Rect(R.x, R.y, R.width, R.height));

	// в итоге найден объект (object), маска, добавили в раскадровку альфа-канал, объект сам с альфа-каналом, написана функция оверлея с 4-м каналом.
	// Время работы с траекторией. Алгоритм простой:
	// Bouning rect объекта (object) у нас находится в R -> x,y,width, height
	// Настало время вычисления по центероидам масс контуров траектории, построения мизансцены и покадровой анимации (восстановление расскадровки, если угодно)

	//траектория
	
	//центероид объекта
	auto mu = cv::moments(g_contours[g_largest_contour_index]);
	auto mc = cv::Point2i(cvRound(mu.m10 / mu.m00), cvRound(mu.m01 / mu.m00));
	auto ulp = g_contours[g_largest_contour_index][0];
	//найдем dx,dy относительно boundingRect контура. Относительные величины.
	auto cdx = mc.x - g_contours[g_largest_contour_index][0].x;
	auto cdy = mc.y - g_contours[g_largest_contour_index][0].y;

	// Теперь будет проще найти центр массы у любого контура

	std::vector<cv::Point> trajectory;

	for (auto const& img : scenes_grey) {
		cv::Mat tmp;
		double largest_area = 0;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::threshold(img, tmp, 100, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU); // средний порог // adaptive for interesting tests?

		cv::findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		int Idx = 0;
		for (int i = 0; i < contours.size(); ++i)
		{
			double area = contourArea(contours[i], false);
			if (area > largest_area) {
				largest_area = area;
				Idx = i;
			}
		}

		// WARN:
		// если в раскадрровке есть контур, пусть и сложный, то объемлющий будет в contours[0]
		// findContours выдает иногда странные баги связанные с координатами, upperleft угол = (0,0), а он выдает (1,1)
		// на других границах он тоже прибавляет пиксель, но внутри кадров считает координаты верно!!!!!!
		// этот баг нужно исправлять opencv, но его природа ясна, на ЛЮБОЙ границе с кадром (у него w,h +1) на границу, внутри все в порядке. (даже для 1 всё контура)
		// нужно попробовать потом opencv 2 последний на gcc, может там его нет.
		if (contours.size())
		{
			auto& outerCountur = contours[Idx];
			// контур может быть за пределами раскадровки, будем искать его координаты
			cv::Rect RO = cv::boundingRect(outerCountur);
			int x(0), y(0);
			if (RO.height != R.height && RO.width != R.width) // углы
			{
				if (RO.x == 0 && RO.y == 0) { // левый вверхний угол
					x = RO.width - R.width;
					y = RO.height - R.height;
				}
				else if (RO.y == 0 && RO.x + R.width > img.cols) { // правый верхний угол
					x = RO.x;
					y = RO.height - R.height;
				}
				else if (RO.x == 0 && RO.y + R.height > img.rows) { // левый нижний угол
					x = RO.width - R.width;
					y = RO.y;
				}
				else if (RO.x + R.width > img.cols && RO.y + R.height > img.rows) { // правый нижний угол
					x = RO.x;
					y = RO.y;
				}
				else { //пытаемся поправить баг, на левой границу он дает нам безобразную "картину"
					// баг 1 пиксела, сюда попадем как раз из-за левой границы
					x = RO.width - R.width;
					if (RO.y == 1)
						y = RO.height - R.height;
					else
						y = RO.y;
				}
			}
			else if (RO.height != R.height && (RO.y == 0 || RO.y < img.rows)) // TOP\BOTTOM + (защ. от неровной раскадровки)
			{
				if (RO.y == 0) { // TOP
					x = RO.x;
					y = RO.height - R.height;
				}
				else { //BOTTOM
					x = RO.x;
					y = RO.y;
				}
			}
			else if (RO.width != R.width && (RO.x == 0 || RO.x < img.cols)) // LEFT\RIGHT + (защ. от неровной раскадровки)
			{
				if (RO.x == 0) { // LEFT
					x = RO.width - R.width;
					y = RO.y;
				}
				else if (RO.x == 0 || RO.x < img.cols){ // RIGHT
					x = RO.x;
					y = RO.y;
				}
				else {
					//пытаемся поправить баг, на левой границу он дает нам безобразную "картину"
					// баг 1 пиксела, сюда попадем как раз из-за левой границы
					x = RO.width - R.width;
					y = RO.y;
				}
			}
			// а может быть и в пределах кадра
			else { // WITHIN
				x = RO.x;
				y = RO.y;
			}

			x += cdx;
			y += cdy;

			trajectory.push_back(cv::Point(x, y));
		}
	}

	if (trajectory.empty()) {
		std::cerr << "При анализе раскадровок не получилось создать траекторию движения объекта. Убедитесь что на кадрах присутствует объект." << std::endl;
		system("pause");
		return -1;
	}

	//построили траекторию в trajectory
	const cv::Point *pts = (const cv::Point*) cv::Mat(trajectory).data;
	int npts = cv::Mat(trajectory).rows;

	//время построения мизансцены

	cv::Mat canvas(scenes[0].size(), scenes[0].type(), background);
	
	//дабы видно было траекторию, нарисуем ее инвертированным цветом
	cv::Scalar trajColor(255 - background.val[2], 255 - background.val[1], 255 - background.val[0]);
	cv::polylines(canvas, &pts, &npts, 1, false, trajColor, /* colour RGB ordering */3, CV_AA, 0);
	std::for_each(trajectory.begin(), trajectory.end(), [&canvas, &trajColor](const cv::Point& pnt){
		circle(canvas, pnt, 5, trajColor, -1, CV_AA, 0);
	});

	//рисуем линию, нужно придумать контрастный цвет..хотя пусть будет тем же инвертированным
	cv::line(canvas, user_line.first, user_line.second, trajColor, 2, CV_AA, 0);

	//мизансцена готова, приступим к вычислениям углов между тректорией и линией если есть хоть одно пересечение, два итд.
	for (int i = 0; i < trajectory.size() - 1; ++i) {
		cv::Point out;
		if (line_intersects(user_line.first, user_line.second, trajectory[i], trajectory[i + 1], &out)) {
			double angle = angleBetween2Lines(user_line.first, user_line.second, trajectory[i], trajectory[i + 1]);
			std::cout << "Найдено пересечение пользовательской линии и траектории под углом: " << angle << " градусов в точке " << out << "." << std::endl;
		}
	}

	//Последний этап - анимация + вычисление максимального сечения
	//анимировать будем рисуя фреймы контура на траектории
	//вычислять позицию центероида на траектории будем алгоритмом Брезенхема
	cv::namedWindow("Macroscop_Task", cv::WINDOW_AUTOSIZE);
	bool stopAnimation = false;
	double maxSection = .0;
	std::cout << std::endl << "Поиск сечений ... " << std::endl;
	for (int i = 0; i < trajectory.size() - 1 && !stopAnimation; ++i) {
		auto& currentLineP1 = trajectory[i];
		auto& currentLineP2 = trajectory[i + 1];

		int x1 = currentLineP1.x, x2 = currentLineP2.x, y1 = currentLineP1.y, y2 = currentLineP2.y;

		const int deltaX = abs(x2 - x1);
		const int deltaY = abs(y2 - y1);
		const int signX = x1 < x2 ? 1 : -1;
		const int signY = y1 < y2 ? 1 : -1;
		
		int error = deltaX - deltaY;

		while (x1 != x2 || y1 != y2)
		{
			cv::Mat frame;
			overlayImage(canvas, object, frame, cv::Point(x1 - cdx, y1 - cdy));
			auto currentSection = findSection(background, object, cv::Point(x1 - cdx, y1 - cdy),
				user_line.first, user_line.second, scenes);
			if (currentSection)
			{
				std::cout << "Найдено сечение: " << currentSection << std::endl;
				if (currentSection > maxSection) { maxSection = currentSection; }
			}
			cv::imshow("Macroscop_Task", frame);
			int c = cv::waitKey(5);
			if (c == 27) {
				stopAnimation = true;
				break;
			}

			const int error2 = error * 2;
			
			if (error2 > -deltaY)
			{
				error -= deltaY;
				x1 += signX;
			}
			if (error2 < deltaX)
			{
				error += deltaX;
				y1 += signY;
			}
		}

		//int dx = currentLineP2.x - currentLineP1.x,
		//	dy = currentLineP2.y - currentLineP1.y,
		//	y = currentLineP1.y,
		//	eps = 0;

		//for (int x = currentLineP1.x; x <= currentLineP2.x; ++x) {
		//	cv::Mat frame;
		//	overlayImage(canvas, object, frame, cv::Point(x - cdx, y - cdy));
		//	cv::imshow("mahMovie", frame);
		//	int c = cv::waitKey(30);
		//	if (c == 27) {
		//		stopAnimation = true;
		//		break;
		//	}
		//	eps += dy;
		//	if ((eps << 1) >= dx) {
		//		y++;
		//		eps -= dx;
		//	}
		//}
	}

	std::cout << std::endl << "Максимальное сечение: " << maxSection << std::endl;

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
