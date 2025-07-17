#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_14_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<14> mm_instantiator;

public:
  UnalignedM_14_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_14_MMInstantiatorWrapper__;

}  // namespace LLMMM
