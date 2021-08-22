using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Diagnostics;


namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }
        private void Window_Loaded(object sender, System.Windows.RoutedEventArgs e)
        {
            var loadImage = Cv2.ImRead("C:/Users/user/work/opencv/test.jpg", ImreadModes.Color);
            var src_bitmap = BitmapSourceConverter.ToBitmapSource(loadImage);
            imgSrc.Source = src_bitmap;

            //HSVカラーに変換
            var hsvImage = loadImage.CvtColor(ColorConversionCodes.BGR2HSV, 3);
            var res = new Mat();
            res = hsvImage.CvtColor(ColorConversionCodes.HSV2BGR, 3);

            //グレースケール変換
            var gray = res.CvtColor(ColorConversionCodes.BGR2GRAY);
            //2値化
            var binImg = gray.Threshold(0, 255, ThresholdTypes.Otsu);

            //画像内の輪郭を抽出
            HierarchyIndex[] hierarchyIndexes;
            Point[][] contours;
            Mat eigenVectors;
            Mat mean;
            binImg.FindContours(out contours, out hierarchyIndexes, RetrievalModes.List, ContourApproximationModes.ApproxNone);
            // img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            Cv2.DrawContours(loadImage, contours, -1, new Scalar(0, 255, 0), 3);
            var dst_bitmap = BitmapSourceConverter.ToBitmapSource(loadImage);
            imgDst.Source = dst_bitmap;

            Mat m;
            for (int i = 0; i < contours.Length; i++)
            {
                double area = Cv2.ContourArea(contours[i]);
                
                if ( area < 100 || area > 20000)
                {
                    continue;
                }
                m = new Mat(contours[i].Length, 2, MatType.CV_32FC1, contours[i]);
                m = m.Reshape(1); // to 1 channel

                // 両方とも空にして初期化
                mean = new Mat();
                eigenVectors = new Mat();
                // 算出
                Cv2.PCACompute(m, mean, eigenVectors, 1);
                Debug.WriteLine(Cv2.Format(mean, FormatType.Python));
                Debug.WriteLine(Cv2.Format(eigenVectors, FormatType.NumPy));

                Point pt = mean.At<Point>(0);
                float x = eigenVectors.At<float>(0, 0);
                float y = eigenVectors.At<float>(0, 1);
                Debug.WriteLine("vec:" + x + "," + y);
                Debug.WriteLine("pt:" + pt);
                Debug.WriteLine("--------------------------");
            }

        }

        /*
        private void drawAxis(img, start_pt, vec):
        {

        }
        */

    }
}