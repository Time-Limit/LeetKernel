#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_25_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<25> mm_instantiator;

public:
  UnalignedM_25_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_25_MMInstantiatorWrapper__;

}  // namespace LLMMM
