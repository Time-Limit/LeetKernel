#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_54_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<54> mm_instantiator;

public:
  UnalignedM_54_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_54_MMInstantiatorWrapper__;

}  // namespace LLMMM
