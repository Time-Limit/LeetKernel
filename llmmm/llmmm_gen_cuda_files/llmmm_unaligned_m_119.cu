#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_119_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<119> mm_instantiator;

public:
  UnalignedM_119_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_119_MMInstantiatorWrapper__;

}  // namespace LLMMM
