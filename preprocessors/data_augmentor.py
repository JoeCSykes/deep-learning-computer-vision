import numpy


class DataAugmentor:
    def __init__(self, augmentation_sequence):
        self.augmentation_sequence = augmentation_sequence

    def _augment_image(self, image, num):
        return [self.augmentation_sequence(image) for _ in range(num)]

    @staticmethod
    def _convert_to_list(image_list):
        shape_of_list = image_list.shape
        if len(shape_of_list) == 3:
            return [image_list]
        if len(shape_of_list) == 4:
            return image_list

    def augment(self, image_list, num_aug_images, label_list=()):

        image_list = self._convert_to_list(image_list)
        aug_images = []
        aug_image_labels = []

        if len(label_list) > 0:
            assert len(image_list) == len(label_list), \
                f"List of images is not the same length as the list of labels."

            for idx in range(len(image_list)):
                aug_images += self._augment_image(image_list[idx], num_aug_images)
                aug_image_labels += [label_list[idx]] * num_aug_images

            numpy_aug_images = numpy.array(aug_images)
            numpy_aug_labels = numpy.array(aug_image_labels)

            return numpy_aug_images, numpy_aug_labels

        else:
            for idx in range(len(image_list)):
                aug_images += self._augment_image(image_list[idx], num_aug_images)

            numpy_aug_images = numpy.array(aug_images)

            return numpy_aug_images
