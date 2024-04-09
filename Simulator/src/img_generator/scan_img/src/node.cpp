#include "scan_image.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "scan_image");
    ScanImage scan_image;
    ros::spin();
    return 0;
}
