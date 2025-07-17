#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_24_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<24> mm_instantiator;

public:
  UnalignedM_24_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_24_MMInstantiatorWrapper__;

}  // namespace LLMMM
