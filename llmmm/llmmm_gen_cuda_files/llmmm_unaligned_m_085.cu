#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_85_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<85> mm_instantiator;

public:
  UnalignedM_85_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_85_MMInstantiatorWrapper__;

}  // namespace LLMMM
