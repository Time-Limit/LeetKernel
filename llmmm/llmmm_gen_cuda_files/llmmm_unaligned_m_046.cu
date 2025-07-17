#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_46_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<46> mm_instantiator;

public:
  UnalignedM_46_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_46_MMInstantiatorWrapper__;

}  // namespace LLMMM
